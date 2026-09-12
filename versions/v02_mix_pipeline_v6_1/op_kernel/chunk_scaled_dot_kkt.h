/*!
 * \file chunk_scaled_dot_kkt.h
 * \brief V6.1 local AIC-to-two-AIV ping-pong slot pipeline.
 */

#ifndef CHUNK_SCALED_DOT_KKT_H
#define CHUNK_SCALED_DOT_KKT_H

#include "kernel_operator.h"

// Run the Matmul object directly on the AIC branch. Keep the Matmul
// implementation in pure-Cube mode; the outer kernel remains MIX 1:2.
#ifndef ASCENDC_CUBE_ONLY
#define ASCENDC_CUBE_ONLY
#endif
#include "lib/matmul_intf.h"
#include "chunk_scaled_dot_kkt_tiling_data.h"

namespace NsChunkScaledDotKkt {

using namespace AscendC;

constexpr int32_t BT = 64;
constexpr int32_t ROW_ELEMS = 64;
constexpr int32_t KKT_ELEMS = BT * BT;
constexpr int32_t KKT_BYTES =
    KKT_ELEMS * static_cast<int32_t>(sizeof(float));
constexpr int32_t SLOT_COUNT = 2;
constexpr int32_t AIVS_PER_AIC = 2;
constexpr int32_t FALLBACK_ROWS_PER_AIV = BT / AIVS_PER_AIC;
constexpr float NEGATIVE_INF_PROXY = -3.402823466e+38F;
constexpr int32_t GATE_HEAD_BATCH = 2;

// Use one READY and one FREE counter per ping/pong slot. Alternating flag IDs
// avoids consecutive Set operations on the same CrossCore counter.
constexpr uint16_t KKT_READY_FLAG_0 = 7;
constexpr uint16_t KKT_READY_FLAG_1 = 8;
constexpr uint16_t KKT_FREE_FLAG_0 = 9;
constexpr uint16_t KKT_FREE_FLAG_1 = 10;

__aicore__ inline uint16_t ReadyFlagForSlot(int32_t slotId)
{
    return slotId == 0 ? KKT_READY_FLAG_0 : KKT_READY_FLAG_1;
}

__aicore__ inline uint16_t FreeFlagForSlot(int32_t slotId)
{
    return slotId == 0 ? KKT_FREE_FLAG_0 : KKT_FREE_FLAG_1;
}

__aicore__ inline int32_t MinI32(int32_t a, int32_t b)
{
    return a < b ? a : b;
}

template <uint32_t K_DIM>
class ChunkScaledDotKktCubeStage {
public:
    using AType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
    using BType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
    using CType = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
    using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
    using MatmulObj = Matmul<AType, BType, CType, BiasType>;

    __aicore__ inline ChunkScaledDotKktCubeStage(MatmulObj& mm)
        : mm_(mm)
    {
    }

    __aicore__ inline void Init(
        GM_ADDR k,
        GM_ADDR chunkOffsets,
        GM_ADDR workspace,
        const ChunkScaledDotKktTilingData* tilingData)
    {
        tiling_ = tilingData;
        kGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(k),
            tiling_->t * tiling_->hg * K_DIM);
        const int64_t numChunks =
            tiling_->totalTaskCount / tiling_->hg;
        chunkOffsetsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(chunkOffsets),
            numChunks + 1);
        kktWorkspaceGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(workspace),
            static_cast<int64_t>(tiling_->blockDim) *
                SLOT_COUNT * KKT_ELEMS);
    }

    __aicore__ inline void Process()
    {
        const int64_t groupId =
            static_cast<int64_t>(GetBlockIdx());
        int32_t localRound = 0;

        for (int64_t taskId = groupId;
             taskId < tiling_->totalTaskCount;
             taskId += static_cast<int64_t>(tiling_->blockDim),
             ++localRound) {
            const int32_t slotId =
                localRound & (SLOT_COUNT - 1);

            // Round r reuses the slot written in round r-2. Mode 2 releases
            // this wait only after both local AIVs have finished reading it.
            if (localRound >= SLOT_COUNT) {
                CrossCoreWaitFlag(
                    FreeFlagForSlot(slotId));
            }

            ProduceKkt(
                taskId,
                groupId,
                slotId);

            // PIPE_FIX orders the READY message after Matmul/Fixpipe has
            // committed the compact KKT tile to this group's GM slot.
            CrossCoreSetFlag<2, PIPE_FIX>(
                ReadyFlagForSlot(slotId));
        }

        mm_.End();
    }

private:
    __aicore__ inline void ProduceKkt(
        int64_t taskId,
        int64_t groupId,
        int32_t slotId)
    {
        const int64_t kvHeadId =
            taskId % tiling_->hg;
        const int64_t chunkId =
            taskId / tiling_->hg;

        const int32_t start =
            chunkOffsetsGm_.GetValue(chunkId);
        const int32_t end =
            chunkOffsetsGm_.GetValue(chunkId + 1);
        const int32_t chunkLen = end - start;

        // The AIC still publishes READY for malformed metadata so that both
        // AIVs can consume the round and preserve the counter protocol.
        if (start < 0 || end > tiling_->t ||
            chunkLen <= 0 || chunkLen > BT) {
            return;
        }

        const int64_t kOffset =
            (static_cast<int64_t>(start) *
                tiling_->hg + kvHeadId) * K_DIM;

        mm_.SetOrgShape(
            chunkLen,
            chunkLen,
            static_cast<int32_t>(
                tiling_->hg * K_DIM),
            static_cast<int32_t>(
                tiling_->hg * K_DIM),
            chunkLen);
        mm_.SetSingleShape(
            chunkLen,
            chunkLen,
            K_DIM);
        mm_.SetTensorA(
            kGm_[kOffset],
            false);
        mm_.SetTensorB(
            kGm_[kOffset],
            true);
        mm_.DisableBias();

        const int64_t slotOffset =
            (groupId * SLOT_COUNT + slotId) *
            static_cast<int64_t>(KKT_ELEMS);
        mm_.template IterateAll<true>(
            kktWorkspaceGm_[slotOffset]);
    }

private:
    MatmulObj& mm_;
    const ChunkScaledDotKktTilingData* tiling_ = nullptr;
    GlobalTensor<bfloat16_t> kGm_;
    GlobalTensor<int32_t> chunkOffsetsGm_;
    GlobalTensor<float> kktWorkspaceGm_;
};

class ChunkScaledDotKktVectorStage {
public:
    __aicore__ inline ChunkScaledDotKktVectorStage(TPipe* pipe)
        : pipe_(pipe)
    {
    }

    __aicore__ inline void Init(
        GM_ADDR beta,
        GM_ADDR gCumsum,
        GM_ADDR chunkOffsets,
        GM_ADDR output,
        GM_ADDR workspace,
        const ChunkScaledDotKktTilingData* tilingData)
    {
        tiling_ = tilingData;

        h_ = tiling_->hg * static_cast<int64_t>(tiling_->numRepeat);
        const int64_t numChunks =
            tiling_->totalTaskCount / tiling_->hg;

        betaGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(beta),
            h_ * tiling_->t);
        gGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(gCumsum),
            h_ * tiling_->t);
        chunkOffsetsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(chunkOffsets),
            numChunks + 1);
        outputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(output),
            h_ * tiling_->t * BT);
        kktWorkspaceGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(workspace),
            static_cast<int64_t>(tiling_->blockDim) *
                SLOT_COUNT * KKT_ELEMS);

        // Head-split tasks use all chunk rows. The numRepeat==1 fallback
        // consumes at most half the rows but shares this fixed UB layout.
        const int32_t tileElems = KKT_ELEMS;
        const int32_t pairTileElems = GATE_HEAD_BATCH * tileElems;
        const int32_t maskBytes = (pairTileElems + 7) / 8;
        const int32_t maskBytesAligned = (maskBytes + 31) & ~31;

        pipe_->InitBuffer(kktBuf_, tileElems * sizeof(float));

        // Keep single/tail and pair paths on independent queues. The original
        // single-head path is sensitive to queue tensor size and shared
        // temporary-space layout; it must not receive a pair-sized tensor.
        pipe_->InitBuffer(
            outputSingleQueue_,
            2,
            tileElems * sizeof(float));
        pipe_->InitBuffer(
            outputPairQueue_,
            2,
            pairTileElems * sizeof(float));
        // Each VECIN queue buffer stores two consecutive heads. One MTE2
        // descriptor therefore replaces two small per-head DMA operations,
        // while the two physical buffers preserve ping-pong prefetching.
        pipe_->InitBuffer(
            gQueue_,
            2,
            GATE_HEAD_BATCH * ROW_ELEMS * sizeof(float));
        pipe_->InitBuffer(
            betaQueue_,
            2,
            GATE_HEAD_BATCH * ROW_ELEMS * sizeof(bfloat16_t));
        // Tail chunks now use gQueue_/betaQueue_ as well. Queue ownership
        // provides the MTE2->Vector dependency explicitly and avoids the
        // manually reused V_MTE2/MTE2_V events of the former TBuf path.
        // Isolate the single/tail mask and scratch from the pair path.
        const int32_t singleMaskBytes = (tileElems + 7) / 8;
        const int32_t singleMaskBytesAligned =
            (singleMaskBytes + 31) & ~31;
        pipe_->InitBuffer(
            negGSingleBuf_,
            ROW_ELEMS * sizeof(float));
        pipe_->InitBuffer(
            negGPairBuf_,
            GATE_HEAD_BATCH * ROW_ELEMS * sizeof(float));
        pipe_->InitBuffer(
            negativeMaskSingleBuf_,
            singleMaskBytesAligned);
        pipe_->InitBuffer(
            negativeMaskPairBuf_,
            maskBytesAligned);

        // Shared pair scratch. Each head reserves 64 row-scalar blocks
        // (64 * 8 FP32 elements), so the same layout supports any rowCount
        // from 1 to 64 without a generic BroadCast temporary buffer.
        pipe_->InitBuffer(
            broadcastScratchBuf_,
            KKT_ELEMS * sizeof(float));

        vToMte2Event_ =
            static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
        mte2ToVEvent_ =
            static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
        pairMte3ToScalarEvent_ =
            static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_S));
    }

    __aicore__ inline void Process()
    {
        const int64_t aivId =
            static_cast<int64_t>(GetBlockIdx());
        const int64_t taskRatio =
            static_cast<int64_t>(GetTaskRation());
        const int64_t groupId =
            aivId / taskRatio;
        const int32_t aivLane =
            static_cast<int32_t>(
                aivId % taskRatio);

        int32_t localRound = 0;
        for (int64_t taskId = groupId;
             taskId < tiling_->totalTaskCount;
             taskId += static_cast<int64_t>(tiling_->blockDim),
             ++localRound) {
            ProcessPipelineTask(
                taskId,
                groupId,
                aivLane,
                localRound);
        }
    }

private:
    __aicore__ inline void ProcessPipelineTask(
        int64_t taskId,
        int64_t groupId,
        int32_t aivLane,
        int32_t localRound)
    {
        const int32_t slotId =
            localRound & (SLOT_COUNT - 1);
        const bool willReuseSlot =
            taskId +
                static_cast<int64_t>(
                    SLOT_COUNT * tiling_->blockDim) <
            tiling_->totalTaskCount;

        const int64_t kvHeadId =
            taskId % tiling_->hg;
        const int64_t chunkId =
            taskId / tiling_->hg;
        const int32_t start =
            chunkOffsetsGm_.GetValue(chunkId);
        const int32_t end =
            chunkOffsetsGm_.GetValue(chunkId + 1);
        const int32_t chunkLen = end - start;

        const bool validChunk =
            start >= 0 &&
            end <= tiling_->t &&
            chunkLen > 0 &&
            chunkLen <= BT;

        int32_t rowStart = 0;
        int32_t rowCount = 0;
        int32_t firstRepeat = 0;
        int32_t repeatEnd = 0;

        if (validChunk) {
            if (tiling_->numRepeat >= AIVS_PER_AIC) {
                // Main path: both AIVs keep complete rows and split the
                // repeated query heads. Both load the same KKT tile.
                rowStart = 0;
                rowCount = chunkLen;
                firstRepeat = static_cast<int32_t>(
                    static_cast<int64_t>(aivLane) *
                    tiling_->numRepeat /
                    AIVS_PER_AIC);
                repeatEnd = static_cast<int32_t>(
                    static_cast<int64_t>(aivLane + 1) *
                    tiling_->numRepeat /
                    AIVS_PER_AIC);
            } else {
                // First synchronization milestone for H/Hg==1: split rows
                // between the two AIVs. The variable-row Brcb path keeps this
                // fully Vectorized. Task staggering is added only after the
                // slot protocol is proven correct.
                firstRepeat = 0;
                repeatEnd = 1;
                rowStart =
                    aivLane * FALLBACK_ROWS_PER_AIV;
                rowCount = MinI32(
                    FALLBACK_ROWS_PER_AIV,
                    chunkLen - rowStart);
                if (rowCount < 0) {
                    rowCount = 0;
                }
            }
        }

        const bool hasWork =
            validChunk &&
            rowCount > 0 &&
            firstRepeat < repeatEnd;

        // Every AIV consumes every READY message, including a lane with no
        // rows or malformed input, so the mode-2 counter remains balanced.
        CrossCoreWaitFlag(
            ReadyFlagForSlot(slotId));

        if (!hasWork) {
            if (willReuseSlot) {
                CrossCoreSetFlag<2, PIPE_MTE2>(
                    FreeFlagForSlot(slotId));
            }
            return;
        }

        LocalTensor<float> kktLocal =
            LoadKktSlot(
                groupId,
                slotId,
                chunkLen,
                rowStart,
                rowCount);

        // Once the compact KKT rows have reached this AIV's private UB, GM
        // slot reuse is safe. Mode 2 releases the AIC only after both AIVs
        // have issued this FREE message.
        if (willReuseSlot) {
            CrossCoreSetFlag<2, PIPE_MTE2>(
                FreeFlagForSlot(slotId));
        }

        ProcessHeadRange(
            kktLocal,
            kvHeadId,
            start,
            chunkLen,
            rowStart,
            rowCount,
            firstRepeat,
            repeatEnd);
    }

    __aicore__ inline void ProcessHeadRange(
        const LocalTensor<float>& kktLocal,
        int64_t kvHeadId,
        int32_t start,
        int32_t chunkLen,
        int32_t rowStart,
        int32_t rowCount,
        int32_t firstRepeat,
        int32_t repeatEnd)
    {
        const int64_t firstHead =
            kvHeadId * tiling_->numRepeat;
        const int32_t totalHeads =
            repeatEnd - firstRepeat;
        const int32_t batchCount =
            (totalHeads + GATE_HEAD_BATCH - 1) /
            GATE_HEAD_BATCH;
        const int32_t firstBatchHeads =
            MinI32(
                GATE_HEAD_BATCH,
                totalHeads);

        PrefetchGateAndBetaBatch(
            firstHead + firstRepeat,
            start,
            chunkLen,
            firstBatchHeads);

        for (int32_t batchId = 0;
             batchId < batchCount;
             ++batchId) {
            const int32_t batchRepeat =
                firstRepeat +
                batchId * GATE_HEAD_BATCH;
            const int32_t currentBatchHeads =
                MinI32(
                    GATE_HEAD_BATCH,
                    repeatEnd - batchRepeat);

            LocalTensor<float> gBatchLocal =
                gQueue_.DeQue<float>();
            LocalTensor<bfloat16_t> betaBatchLocal =
                betaQueue_.DeQue<bfloat16_t>();

            if (batchId + 1 < batchCount) {
                const int32_t nextBatchRepeat =
                    batchRepeat +
                    GATE_HEAD_BATCH;
                const int32_t nextBatchHeads =
                    MinI32(
                        GATE_HEAD_BATCH,
                        repeatEnd -
                            nextBatchRepeat);
                PrefetchGateAndBetaBatch(
                    firstHead + nextBatchRepeat,
                    start,
                    chunkLen,
                    nextBatchHeads);
            }

            if (chunkLen != ROW_ELEMS) {
                ZeroTailGateAndBetaBatchSuffix(
                    gBatchLocal,
                    betaBatchLocal,
                    chunkLen,
                    currentBatchHeads);
            }

            if (currentBatchHeads ==
                GATE_HEAD_BATCH) {
                ProcessHeadPairTile(
                    kktLocal,
                    gBatchLocal,
                    betaBatchLocal,
                    firstHead + batchRepeat,
                    start,
                    rowStart,
                    rowCount);
            } else {
                ProcessHeadTile(
                    kktLocal,
                    gBatchLocal,
                    betaBatchLocal,
                    firstHead + batchRepeat,
                    start,
                    rowStart,
                    rowCount);
            }

            gQueue_.FreeTensor(
                gBatchLocal);
            betaQueue_.FreeTensor(
                betaBatchLocal);
        }
    }

    __aicore__ inline LocalTensor<float> LoadKktSlot(
        int64_t groupId,
        int32_t slotId,
        int32_t chunkLen,
        int32_t rowStart,
        int32_t rowCount)
    {
        LocalTensor<float> kktLocal = kktBuf_.Get<float>();

        // A complete 64-column chunk fully overwrites every active UB row.
        // Only a tail chunk needs zero initialization for the columns that
        // DataCopyPad does not cover.
        if (chunkLen != ROW_ELEMS) {
            Duplicate(
                kktLocal,
                0.0f,
                rowCount * ROW_ELEMS);
            PipeBarrier<PIPE_V>();
        }

        // kktBuf_ is a reusable TBuf. Ensure the previous Vector reads have
        // completed before MTE2 overwrites the same UB storage.
        SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
        WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);

        const int64_t slotOffset =
            (groupId * SLOT_COUNT + slotId) *
            static_cast<int64_t>(KKT_ELEMS);
        GlobalTensor<float> taskKkt =
            kktWorkspaceGm_[slotOffset];
        if (chunkLen == ROW_ELEMS) {
            DataCopy(
                kktLocal,
                taskKkt[rowStart * ROW_ELEMS],
                static_cast<uint32_t>(rowCount * ROW_ELEMS));
        } else {
            const int32_t alignedRowElems =
                ((chunkLen + 7) / 8) * 8;
            const uint8_t rightPad = static_cast<uint8_t>(
                alignedRowElems - chunkLen);
            // For a LocalTensor destination, dstStride is measured in
            // 32-byte data blocks, not bytes. DataCopyPad writes one aligned
            // row and then skips to the next 64-element UB row.
            const uint32_t dstStrideBlocks = static_cast<uint32_t>(
                (ROW_ELEMS - alignedRowElems) / 8);
            DataCopyExtParams copyParams{
                static_cast<uint16_t>(rowCount),
                static_cast<uint32_t>(chunkLen * sizeof(float)),
                0,
                dstStrideBlocks,
                0};
            DataCopyPadExtParams<float> padParams{
                true, 0, rightPad, 0.0f};
            DataCopyPad(
                kktLocal,
                taskKkt[rowStart * chunkLen],
                copyParams,
                padParams);
        }

        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        return kktLocal;
    }

    __aicore__ inline void PrefetchGateAndBetaBatch(
        int64_t firstHeadId,
        int32_t start,
        int32_t chunkLen,
        int32_t batchHeads)
    {
        LocalTensor<float> gBatchLocal =
            gQueue_.AllocTensor<float>();
        LocalTensor<bfloat16_t> betaBatchLocal =
            betaQueue_.AllocTensor<bfloat16_t>();

        const int64_t firstHeadOffset =
            firstHeadId * tiling_->t + start;

        if (chunkLen == ROW_ELEMS) {
            // Full chunks need no destination padding.
            if (batchHeads == 1) {
                DataCopy(
                    gBatchLocal,
                    gGm_[firstHeadOffset],
                    ROW_ELEMS);
                DataCopy(
                    betaBatchLocal,
                    betaGm_[firstHeadOffset],
                    ROW_ELEMS);
            } else {
                const uint32_t gSrcStrideBytes =
                    static_cast<uint32_t>(
                        (tiling_->t - ROW_ELEMS) *
                        sizeof(float));
                const uint32_t betaSrcStrideBytes =
                    static_cast<uint32_t>(
                        (tiling_->t - ROW_ELEMS) *
                        sizeof(bfloat16_t));

                DataCopyExtParams gCopyParams{
                    static_cast<uint16_t>(batchHeads),
                    static_cast<uint32_t>(
                        ROW_ELEMS * sizeof(float)),
                    gSrcStrideBytes,
                    0,
                    0};
                DataCopyPadExtParams<float> gPadParams{
                    false, 0, 0, 0.0f};
                DataCopyPad(
                    gBatchLocal,
                    gGm_[firstHeadOffset],
                    gCopyParams,
                    gPadParams);

                DataCopyExtParams betaCopyParams{
                    static_cast<uint16_t>(batchHeads),
                    static_cast<uint32_t>(
                        ROW_ELEMS * sizeof(bfloat16_t)),
                    betaSrcStrideBytes,
                    0,
                    0};
                DataCopyPadExtParams<bfloat16_t> betaPadParams{
                    false, 0, 0, ToBfloat16(0.0f)};
                DataCopyPad(
                    betaBatchLocal,
                    betaGm_[firstHeadOffset],
                    betaCopyParams,
                    betaPadParams);
            }
        } else {
            // Tail heads are stored compactly in GM but at a fixed
            // [head0:64][head1:64] stride in UB. DataCopyPad fills through
            // the next 32-byte boundary; the remaining suffix is zeroed
            // after DeQue by ZeroTailGateAndBetaBatchSuffix().
            const int32_t gAlignedElems =
                ((chunkLen + 7) / 8) * 8;
            const int32_t betaAlignedElems =
                ((chunkLen + 15) / 16) * 16;

            const uint32_t gSrcStrideBytes =
                static_cast<uint32_t>(
                    (tiling_->t - chunkLen) *
                    sizeof(float));
            const uint32_t betaSrcStrideBytes =
                static_cast<uint32_t>(
                    (tiling_->t - chunkLen) *
                    sizeof(bfloat16_t));
            const uint32_t gDstStrideBlocks =
                static_cast<uint32_t>(
                    (ROW_ELEMS - gAlignedElems) / 8);
            const uint32_t betaDstStrideBlocks =
                static_cast<uint32_t>(
                    (ROW_ELEMS - betaAlignedElems) / 16);

            DataCopyExtParams gCopyParams{
                static_cast<uint16_t>(batchHeads),
                static_cast<uint32_t>(
                    chunkLen * sizeof(float)),
                gSrcStrideBytes,
                gDstStrideBlocks,
                0};
            DataCopyPadExtParams<float> gPadParams{
                true,
                0,
                static_cast<uint8_t>(
                    gAlignedElems - chunkLen),
                0.0f};
            DataCopyPad(
                gBatchLocal,
                gGm_[firstHeadOffset],
                gCopyParams,
                gPadParams);

            DataCopyExtParams betaCopyParams{
                static_cast<uint16_t>(batchHeads),
                static_cast<uint32_t>(
                    chunkLen * sizeof(bfloat16_t)),
                betaSrcStrideBytes,
                betaDstStrideBlocks,
                0};
            DataCopyPadExtParams<bfloat16_t> betaPadParams{
                true,
                0,
                static_cast<uint8_t>(
                    betaAlignedElems - chunkLen),
                ToBfloat16(0.0f)};
            DataCopyPad(
                betaBatchLocal,
                betaGm_[firstHeadOffset],
                betaCopyParams,
                betaPadParams);
        }

        gQueue_.EnQue(gBatchLocal);
        betaQueue_.EnQue(betaBatchLocal);
    }

    __aicore__ inline void ZeroTailGateAndBetaBatchSuffix(
        LocalTensor<float> gBatchLocal,
        LocalTensor<bfloat16_t> betaBatchLocal,
        int32_t chunkLen,
        int32_t batchHeads)
    {
        const int32_t gAlignedElems =
            ((chunkLen + 7) / 8) * 8;
        const int32_t betaAlignedElems =
            ((chunkLen + 15) / 16) * 16;

        bool issuedVector = false;
        for (int32_t headInBatch = 0;
             headInBatch < batchHeads;
             ++headInBatch) {
            const int32_t headOffset =
                headInBatch * ROW_ELEMS;

            if (gAlignedElems < ROW_ELEMS) {
                Duplicate(
                    gBatchLocal[
                        headOffset + gAlignedElems],
                    0.0f,
                    ROW_ELEMS - gAlignedElems);
                issuedVector = true;
            }
            if (betaAlignedElems < ROW_ELEMS) {
                Duplicate(
                    betaBatchLocal[
                        headOffset + betaAlignedElems],
                    ToBfloat16(0.0f),
                    ROW_ELEMS - betaAlignedElems);
                issuedVector = true;
            }
        }

        if (issuedVector) {
            PipeBarrier<PIPE_V>();
        }
    }


    __aicore__ inline void ProcessHeadPairTile(
        const LocalTensor<float>& kktLocal,
        const LocalTensor<float>& gPairLocal,
        const LocalTensor<bfloat16_t>& betaPairLocal,
        int64_t firstHeadId,
        int32_t start,
        int32_t rowStart,
        int32_t rowCount)
    {
        LocalTensor<float> negGPairLocal =
            negGPairBuf_.Get<float>();
        LocalTensor<float> scratch =
            broadcastScratchBuf_.Get<float>();
        LocalTensor<uint8_t> negativeMask =
            negativeMaskPairBuf_.Get<uint8_t>();
        LocalTensor<float> outputPairLocal =
            outputPairQueue_.AllocTensor<float>();

        const int32_t activeElems =
            rowCount * ROW_ELEMS;
        const int32_t pairActiveElems =
            GATE_HEAD_BATCH * activeElems;
        // Brcb requires a 32-byte-aligned source address. rowStart may be
        // smaller than eight when rowTileCount is large, so align the source
        // down to an 8-FP32 boundary, materialize a short prefix, and skip
        // that prefix in the aligned scratch tensor.
        const int32_t alignedRowStart =
            rowStart & ~7;
        const int32_t rowPrefix =
            rowStart - alignedRowStart;
        const uint8_t rowBrcbRepeats =
            static_cast<uint8_t>(
                (rowPrefix + rowCount + 7) / 8);
        const int32_t scratchRowOffset =
            rowPrefix * 8;
        constexpr int32_t scratchHeadStride =
            ROW_ELEMS * 8;

        // Both -g_j column vectors remain full 64 elements. Tail suffixes
        // were zero-filled after DeQue, matching the proven tail behavior.
        Muls(
            negGPairLocal,
            gPairLocal,
            -1.0f,
            GATE_HEAD_BATCH * ROW_ELEMS);
        PipeBarrier<PIPE_V>();

        // Materialize only the active row scalars. Brcb may read up to seven
        // padded rows at the end of a repeat, but Add consumes exactly
        // rowCount repeats.
        if (rowStart == 0 && rowCount == ROW_ELEMS) {
            // Preserve V5.14's lowest-overhead full64 instruction: both
            // contiguous heads are collected by one 16-repeat Brcb.
            Brcb(
                scratch,
                gPairLocal,
                static_cast<uint8_t>(
                    GATE_HEAD_BATCH * ROW_ELEMS / 8),
                {1, 8});
        } else {
            Brcb(
                scratch,
                gPairLocal[alignedRowStart],
                rowBrcbRepeats,
                {1, 8});
            Brcb(
                scratch[scratchHeadStride],
                gPairLocal[ROW_ELEMS + alignedRowStart],
                rowBrcbRepeats,
                {1, 8});
        }
        PipeBarrier<PIPE_V>();

        Add(
            outputPairLocal,
            negGPairLocal,
            scratch[scratchRowOffset],
            static_cast<uint64_t>(ROW_ELEMS),
            static_cast<uint8_t>(rowCount),
            {1, 1, 0, 8, 0, 1});
        Add(
            outputPairLocal[activeElems],
            negGPairLocal[ROW_ELEMS],
            scratch[scratchHeadStride + scratchRowOffset],
            static_cast<uint64_t>(ROW_ELEMS),
            static_cast<uint8_t>(rowCount),
            {1, 1, 0, 8, 0, 1});
        PipeBarrier<PIPE_V>();

        CompareScalar(
            negativeMask,
            outputPairLocal,
            0.0f,
            CMPMODE::LT,
            pairActiveElems);
        PipeBarrier<PIPE_V>();
        Select(
            outputPairLocal,
            negativeMask,
            outputPairLocal,
            NEGATIVE_INF_PROXY,
            SELMODE::VSEL_TENSOR_SCALAR_MODE,
            pairActiveElems);
        PipeBarrier<PIPE_V>();
        Exp(
            outputPairLocal,
            outputPairLocal,
            pairActiveElems);
        PipeBarrier<PIPE_V>();

        // Convert the compact two-head beta vectors once, then Brcb only the
        // active row ranges. No Scalar GetValue or S_V event remains in the
        // complete pair path.
        Cast(
            negGPairLocal,
            betaPairLocal,
            RoundMode::CAST_NONE,
            GATE_HEAD_BATCH * ROW_ELEMS);
        PipeBarrier<PIPE_V>();

        if (rowStart == 0 && rowCount == ROW_ELEMS) {
            Brcb(
                scratch,
                negGPairLocal,
                static_cast<uint8_t>(
                    GATE_HEAD_BATCH * ROW_ELEMS / 8),
                {1, 8});
        } else {
            Brcb(
                scratch,
                negGPairLocal[alignedRowStart],
                rowBrcbRepeats,
                {1, 8});
            Brcb(
                scratch[scratchHeadStride],
                negGPairLocal[ROW_ELEMS + alignedRowStart],
                rowBrcbRepeats,
                {1, 8});
        }
        PipeBarrier<PIPE_V>();

        Mul(
            outputPairLocal,
            outputPairLocal,
            scratch[scratchRowOffset],
            static_cast<uint64_t>(ROW_ELEMS),
            static_cast<uint8_t>(rowCount),
            {1, 1, 0, 8, 8, 1});
        Mul(
            outputPairLocal[activeElems],
            outputPairLocal[activeElems],
            scratch[scratchHeadStride + scratchRowOffset],
            static_cast<uint64_t>(ROW_ELEMS),
            static_cast<uint8_t>(rowCount),
            {1, 1, 0, 8, 8, 1});
        PipeBarrier<PIPE_V>();

        Mul(
            outputPairLocal,
            kktLocal,
            outputPairLocal,
            activeElems);
        Mul(
            outputPairLocal[activeElems],
            kktLocal,
            outputPairLocal[activeElems],
            activeElems);
        PipeBarrier<PIPE_V>();

        outputPairQueue_.EnQue(outputPairLocal);
        CopyOutHeadPair(
            firstHeadId,
            start,
            rowStart,
            rowCount,
            activeElems);
    }

    __aicore__ inline void ProcessHeadTile(
        const LocalTensor<float>& kktLocal,
        const LocalTensor<float>& gLocal,
        const LocalTensor<bfloat16_t>& betaLocal,
        int64_t headId,
        int32_t start,
        int32_t rowStart,
        int32_t rowCount)
    {
        LocalTensor<float> negGLocal =
            negGSingleBuf_.Get<float>();
        LocalTensor<float> scratch =
            broadcastScratchBuf_.Get<float>();
        LocalTensor<uint8_t> negativeMask =
            negativeMaskSingleBuf_.Get<uint8_t>();
        LocalTensor<float> outputLocal =
            outputSingleQueue_.AllocTensor<float>();

        const int32_t activeElems =
            rowCount * ROW_ELEMS;

        // Brcb source addresses must be 32-byte aligned. Align rowStart down
        // to an 8-FP32 boundary, then skip the materialized prefix in scratch.
        const int32_t alignedRowStart =
            rowStart & ~7;
        const int32_t rowPrefix =
            rowStart - alignedRowStart;
        const uint8_t rowBrcbRepeats =
            static_cast<uint8_t>(
                (rowPrefix + rowCount + 7) / 8);
        const int32_t scratchRowOffset =
            rowPrefix * 8;

        Muls(
            negGLocal,
            gLocal,
            -1.0f,
            ROW_ELEMS);
        PipeBarrier<PIPE_V>();

        // Materialize only the active g_i row scalars and reuse the same
        // -g_j[64] vector for every output row.
        Brcb(
            scratch,
            gLocal[alignedRowStart],
            rowBrcbRepeats,
            {1, 8});
        PipeBarrier<PIPE_V>();

        Add(
            outputLocal,
            negGLocal,
            scratch[scratchRowOffset],
            static_cast<uint64_t>(ROW_ELEMS),
            static_cast<uint8_t>(rowCount),
            {1, 1, 0, 8, 0, 1});
        PipeBarrier<PIPE_V>();

        CompareScalar(
            negativeMask,
            outputLocal,
            0.0f,
            CMPMODE::LT,
            activeElems);
        PipeBarrier<PIPE_V>();

        Select(
            outputLocal,
            negativeMask,
            outputLocal,
            NEGATIVE_INF_PROXY,
            SELMODE::VSEL_TENSOR_SCALAR_MODE,
            activeElems);
        PipeBarrier<PIPE_V>();

        Exp(
            outputLocal,
            outputLocal,
            activeElems);
        PipeBarrier<PIPE_V>();

        // Reuse negGLocal as the compact FP32 beta tensor.
        Cast(
            negGLocal,
            betaLocal,
            RoundMode::CAST_NONE,
            ROW_ELEMS);
        PipeBarrier<PIPE_V>();

        Brcb(
            scratch,
            negGLocal[alignedRowStart],
            rowBrcbRepeats,
            {1, 8});
        PipeBarrier<PIPE_V>();

        Mul(
            outputLocal,
            outputLocal,
            scratch[scratchRowOffset],
            static_cast<uint64_t>(ROW_ELEMS),
            static_cast<uint8_t>(rowCount),
            {1, 1, 0, 8, 8, 1});
        PipeBarrier<PIPE_V>();

        Mul(
            outputLocal,
            kktLocal,
            outputLocal,
            activeElems);
        PipeBarrier<PIPE_V>();

        outputSingleQueue_.EnQue(outputLocal);
        CopyOutTile(
            headId,
            start,
            rowStart,
            rowCount);
    }


    __aicore__ inline void CopyOutHeadPair(
        int64_t firstHeadId,
        int32_t start,
        int32_t rowStart,
        int32_t rowCount,
        int32_t activeElems)
    {
        LocalTensor<float> outputPairLocal =
            outputPairQueue_.DeQue<float>();

        const int64_t firstOutputOffset =
            (firstHeadId * tiling_->t + start + rowStart) *
            static_cast<int64_t>(ROW_ELEMS);
        const int64_t secondOutputOffset =
            ((firstHeadId + 1) * tiling_->t + start + rowStart) *
            static_cast<int64_t>(ROW_ELEMS);

        // Keep two conventional MTE3 copies for the first implementation.
        // This avoids introducing direction-specific DataCopyPad stride rules
        // while still removing one queue allocation/transition per head pair.
        DataCopy(
            outputGm_[firstOutputOffset],
            outputPairLocal,
            static_cast<uint32_t>(activeElems));
        DataCopy(
            outputGm_[secondOutputOffset],
            outputPairLocal[activeElems],
            static_cast<uint32_t>(activeElems));

        // Probe 2: wait on the Scalar pipeline until both MTE3 copies,
        // issued in order above, have completed. This deliberately removes
        // any chance that FreeTensor/AllocTensor reuses the pair buffer early.
        SetFlag<HardEvent::MTE3_S>(pairMte3ToScalarEvent_);
        WaitFlag<HardEvent::MTE3_S>(pairMte3ToScalarEvent_);

        outputPairQueue_.FreeTensor(outputPairLocal);
    }

    __aicore__ inline void CopyOutTile(
        int64_t headId,
        int32_t start,
        int32_t rowStart,
        int32_t rowCount)
    {
        LocalTensor<float> outputLocal = outputSingleQueue_.DeQue<float>();
        const int64_t outputOffset =
            (headId * tiling_->t + start + rowStart) *
            static_cast<int64_t>(ROW_ELEMS);
        DataCopy(
            outputGm_[outputOffset],
            outputLocal,
            static_cast<uint32_t>(rowCount * ROW_ELEMS));

        // The Vectorized single-head path can finish quickly enough for the
        // VECOUT buffer to be recycled before MTE3 has consumed it. The pair
        // path already requires the same explicit lifetime fence. Reuse that
        // event here because pair and single CopyOut execute serially on one
        // AIV.
        SetFlag<HardEvent::MTE3_S>(pairMte3ToScalarEvent_);
        WaitFlag<HardEvent::MTE3_S>(pairMte3ToScalarEvent_);

        outputSingleQueue_.FreeTensor(outputLocal);
    }

private:
    TPipe* pipe_ = nullptr;
    const ChunkScaledDotKktTilingData* tiling_ = nullptr;

    int64_t h_ = 0;
    GlobalTensor<bfloat16_t> betaGm_;
    GlobalTensor<float> gGm_;
    GlobalTensor<int32_t> chunkOffsetsGm_;
    GlobalTensor<float> outputGm_;
    GlobalTensor<float> kktWorkspaceGm_;

    TBuf<TPosition::VECIN> kktBuf_;
    TQue<TPosition::VECOUT, 1> outputSingleQueue_;
    TQue<TPosition::VECOUT, 1> outputPairQueue_;
    TQue<TPosition::VECIN, 1> gQueue_;
    TQue<TPosition::VECIN, 1> betaQueue_;
    TBuf<TPosition::VECCALC> negGSingleBuf_;
    TBuf<TPosition::VECCALC> negGPairBuf_;
    TBuf<TPosition::VECCALC> negativeMaskSingleBuf_;
    TBuf<TPosition::VECCALC> negativeMaskPairBuf_;
    TBuf<TPosition::VECCALC> broadcastScratchBuf_;

    event_t vToMte2Event_;
    event_t mte2ToVEvent_;
    event_t pairMte3ToScalarEvent_;
};

} // namespace NsChunkScaledDotKkt

#endif // CHUNK_SCALED_DOT_KKT_