/*!
 * \file chunk_scaled_dot_kkt.h
 * \brief Packed direct-MMAD implementation for K=128 and K=256.
 */

#ifndef CHUNK_SCALED_DOT_KKT_H
#define CHUNK_SCALED_DOT_KKT_H

#include "kernel_operator.h"

#include "chunk_scaled_dot_kkt_tiling_data.h"

// Ascend 910B / Atlas A2 maps to NPU architecture 2201. Keep only this
// instruction-set path to reduce parser work and prevent accidental builds
// for an unvalidated device architecture.
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 2201)
#error "ChunkScaledDotKkt supports Ascend 910B / NPU architecture 2201 only"
#endif

namespace NsChunkScaledDotKkt {

using namespace AscendC;

constexpr int32_t BT = 64;
constexpr int32_t ROW_ELEMS = 64;
constexpr int32_t KKT_ELEMS = BT * BT;
constexpr int32_t SLOT_COUNT = 2;
constexpr int32_t AIVS_PER_AIC = 2;
constexpr int32_t FALLBACK_ROWS_PER_AIV = BT / AIVS_PER_AIC;
constexpr float NEGATIVE_INF_PROXY = -3.402823466e+38F;
constexpr int32_t GATE_HEAD_BATCH = 2;
constexpr int32_t OUTPUT_RETIRE_BATCHES = 4;
constexpr int32_t BROADCAST_SCRATCH_ELEMS =
    GATE_HEAD_BATCH * ROW_ELEMS * 8;

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

class PackedGramMmadCubeStageK256 {
    // K=256 conservative V2:
    //   * four physical L0A/L0B stages, one per 64-wide K-part;
    //   * no K_PART template instantiation;
    //   * no per-stage M_MTE1 counters;
    //   * one task-local M->MTE1 drain after all four MMADs.
public:
    static constexpr uint32_t K_DIM = 256U;
    static constexpr uint32_t M = 64U;
    static constexpr uint32_t N = 64U;
    static constexpr uint32_t K_STEP = 64U;
    static constexpr uint32_t K_PARTS = 4U;
    static constexpr uint32_t FRACTAL_M = 16U;
    static constexpr uint32_t FRACTAL_K = 16U;
    static constexpr uint32_t N_BLOCKS = N / FRACTAL_M;

    static constexpr uint32_t A1_ELEMS = M * K_DIM;
    static constexpr uint32_t B1_ELEMS = N * K_DIM;
    static constexpr uint32_t A2_ELEMS = M * K_STEP;
    static constexpr uint32_t B2_ELEMS = K_STEP * N;
    static constexpr uint32_t CO1_ELEMS = M * N;

    // A and B are the same K tile. Their Nd2Nz packing parameters are
    // identical, so both logical operands alias one physical L1 region.
    static constexpr uint32_t SHARED_L1_ADDR = 0U;
    static constexpr uint32_t SHARED_L1_BYTES =
        A1_ELEMS * sizeof(bfloat16_t);

    static constexpr uint32_t L0A_CAPACITY_BYTES = 64U * 1024U;
    static constexpr uint32_t L0B_CAPACITY_BYTES = 64U * 1024U;
    static constexpr uint32_t L0_STAGE_BYTES = 16U * 1024U;
    static constexpr uint32_t A2_BYTES =
        A2_ELEMS * sizeof(bfloat16_t);
    static constexpr uint32_t B2_BYTES =
        B2_ELEMS * sizeof(bfloat16_t);

    static constexpr uint32_t A2_ADDR0 = 0U * L0_STAGE_BYTES;
    static constexpr uint32_t A2_ADDR1 = 1U * L0_STAGE_BYTES;
    static constexpr uint32_t A2_ADDR2 = 2U * L0_STAGE_BYTES;
    static constexpr uint32_t A2_ADDR3 = 3U * L0_STAGE_BYTES;
    static constexpr uint32_t B2_ADDR0 = 0U * L0_STAGE_BYTES;
    static constexpr uint32_t B2_ADDR1 = 1U * L0_STAGE_BYTES;
    static constexpr uint32_t B2_ADDR2 = 2U * L0_STAGE_BYTES;
    static constexpr uint32_t B2_ADDR3 = 3U * L0_STAGE_BYTES;
    static constexpr uint32_t CO1_ADDR = 0U;

    // One ownership token is sufficient: both L0A and L0B loads are issued
    // on PIPE_MTE1, and the token is returned only after the final B read.
    static constexpr event_t SHARED_L1_EVENT = EVENT_ID0;
    static constexpr event_t MTE1_TO_M_EVENT = EVENT_ID0;
    static constexpr event_t TASK_L0_DRAIN_EVENT = EVENT_ID0;
    static constexpr event_t CO1_EVENT = EVENT_ID0;

    static_assert(A1_ELEMS == B1_ELEMS,
        "A1/B1 aliases require identical packed tile sizes");
    static_assert(SHARED_L1_ADDR % 32U == 0U,
        "Shared L1 base address must be 32-byte aligned");
    static_assert(SHARED_L1_BYTES % 32U == 0U,
        "Shared L1 tile must be 32-byte aligned");
    static_assert(SHARED_L1_ADDR + SHARED_L1_BYTES <= 512U * 1024U,
        "Shared L1 tile exceeds Atlas A2 L1 capacity");
    static_assert(A2_BYTES <= L0_STAGE_BYTES,
        "A2 tile exceeds one L0A stage");
    static_assert(B2_BYTES <= L0_STAGE_BYTES,
        "B2 tile exceeds one L0B stage");
    static_assert(A2_ADDR3 + A2_BYTES <= L0A_CAPACITY_BYTES,
        "Four A2 stages exceed L0A capacity");
    static_assert(B2_ADDR3 + B2_BYTES <= L0B_CAPACITY_BYTES,
        "Four B2 stages exceed L0B capacity");
    static_assert((CO1_ELEMS * sizeof(float)) % 1024U == 0U,
        "CO1 tile must be 1024-byte aligned");

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
        InitSocState();

        const int64_t groupId =
            static_cast<int64_t>(GetBlockIdx());
        int32_t localRound = 0;

        // A1/B1 are two typed views of the same physical L1 bytes.
        LocalTensor<bfloat16_t> a1Local(
            TPosition::A1, SHARED_L1_ADDR, A1_ELEMS);
        LocalTensor<bfloat16_t> b1Local(
            TPosition::B1, SHARED_L1_ADDR, B1_ELEMS);
        LocalTensor<bfloat16_t> a2Local0(
            TPosition::A2, A2_ADDR0, A2_ELEMS);
        LocalTensor<bfloat16_t> a2Local1(
            TPosition::A2, A2_ADDR1, A2_ELEMS);
        LocalTensor<bfloat16_t> a2Local2(
            TPosition::A2, A2_ADDR2, A2_ELEMS);
        LocalTensor<bfloat16_t> a2Local3(
            TPosition::A2, A2_ADDR3, A2_ELEMS);
        LocalTensor<bfloat16_t> b2Local0(
            TPosition::B2, B2_ADDR0, B2_ELEMS);
        LocalTensor<bfloat16_t> b2Local1(
            TPosition::B2, B2_ADDR1, B2_ELEMS);
        LocalTensor<bfloat16_t> b2Local2(
            TPosition::B2, B2_ADDR2, B2_ELEMS);
        LocalTensor<bfloat16_t> b2Local3(
            TPosition::B2, B2_ADDR3, B2_ELEMS);
        LocalTensor<float> c1Local(
            TPosition::CO1, CO1_ADDR, CO1_ELEMS);

        InitPersistentEvents();

        for (int64_t taskId = groupId;
             taskId < tiling_->totalTaskCount;
             taskId += static_cast<int64_t>(tiling_->blockDim),
             ++localRound) {
            const int32_t slotId =
                localRound & (SLOT_COUNT - 1);
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
                chunkLen <= static_cast<int32_t>(M);

            if (validChunk) {
                const int64_t kOffset =
                    (static_cast<int64_t>(start) *
                        tiling_->hg + kvHeadId) * K_DIM;

                CopyPackedKToSharedL1(
                    a1Local,
                    kGm_[kOffset],
                    chunkLen);
                ComputeGram(
                    a1Local,
                    b1Local,
                    a2Local0,
                    a2Local1,
                    a2Local2,
                    a2Local3,
                    b2Local0,
                    b2Local1,
                    b2Local2,
                    b2Local3,
                    c1Local);
            }

            if (localRound >= SLOT_COUNT) {
                CrossCoreWaitFlag(
                    FreeFlagForSlot(slotId));
            }

            if (validChunk) {
                CommitWorkspace(
                    c1Local,
                    groupId,
                    slotId);
            }

            CrossCoreSetFlag<2, PIPE_FIX>(
                ReadyFlagForSlot(slotId));

            if (validChunk) {
                SetFlag<HardEvent::FIX_M>(CO1_EVENT);
            }
        }

        DrainPersistentEvents();
    }

private:
    __aicore__ inline void InitPersistentEvents()
    {
        SetFlag<HardEvent::MTE1_MTE2>(SHARED_L1_EVENT);
        SetFlag<HardEvent::FIX_M>(CO1_EVENT);
    }

    __aicore__ inline void DrainPersistentEvents()
    {
        WaitFlag<HardEvent::MTE1_MTE2>(SHARED_L1_EVENT);
        WaitFlag<HardEvent::FIX_M>(CO1_EVENT);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ClearSharedL1(
        const LocalTensor<bfloat16_t>& sharedL1Local)
    {
        constexpr uint16_t sharedBlocks =
            static_cast<uint16_t>(SHARED_L1_BYTES / 32U);

        InitConstValue(
            sharedL1Local,
            {1, sharedBlocks, 0, ToBfloat16(0.0f)});
    }

    __aicore__ inline void CopyOnePackedOperand(
        const LocalTensor<bfloat16_t>& dstLocal,
        const GlobalTensor<bfloat16_t>& srcGlobal,
        int32_t chunkLen)
    {
        Nd2NzParams copyParams{};
        copyParams.ndNum = 1;
        copyParams.nValue =
            static_cast<uint16_t>(chunkLen);
        copyParams.dValue =
            static_cast<uint16_t>(K_DIM);
        copyParams.srcNdMatrixStride = 0;
        copyParams.srcDValue =
            static_cast<uint16_t>(tiling_->hg * K_DIM);
        copyParams.dstNzC0Stride =
            static_cast<uint16_t>(M);
        copyParams.dstNzNStride = 1;
        copyParams.dstNzMatrixStride = 0;
        DataCopy(dstLocal, srcGlobal, copyParams);
    }

    __aicore__ inline void CopyPackedKToSharedL1(
        const LocalTensor<bfloat16_t>& sharedL1Local,
        const GlobalTensor<bfloat16_t>& srcGlobal,
        int32_t chunkLen)
    {
        WaitFlag<HardEvent::MTE1_MTE2>(SHARED_L1_EVENT);

        if (chunkLen < static_cast<int32_t>(M)) {
            ClearSharedL1(sharedL1Local);
        }

        // One GM->L1 transfer feeds both logical operands.
        CopyOnePackedOperand(sharedL1Local, srcGlobal, chunkLen);
        SetFlag<HardEvent::MTE2_MTE1>(SHARED_L1_EVENT);
    }

    __aicore__ inline void LoadKPartToL0A(
        const LocalTensor<bfloat16_t>& a2Local,
        const LocalTensor<bfloat16_t>& a1Local,
        uint32_t kPart)
    {
        const uint32_t srcAddr =
            kPart * K_STEP * M;

        LoadData3DParamsV2<bfloat16_t> params{};
        params.l1H = 1;
        params.l1W = M;
        params.channelSize = K_STEP;
        params.kExtension = K_STEP;
        params.mExtension = M;
        params.mStartPt = 0;
        params.kStartPt = 0;
        LoadData(a2Local, a1Local[srcAddr], params);
    }

    __aicore__ inline void LoadKPartToL0B(
        const LocalTensor<bfloat16_t>& b2Local,
        const LocalTensor<bfloat16_t>& b1Local,
        uint32_t kPart)
    {
        const uint32_t srcAddr =
            kPart * K_STEP * N;

        LoadData2DParams params{};
        params.repeatTimes =
            static_cast<uint8_t>(N_BLOCKS);
        params.srcStride = 1;
        params.dstGap = 0;
        params.ifTranspose = false;

        constexpr uint32_t blockOffset =
            N * FRACTAL_K;

        LoadData(b2Local[0U * blockOffset],
            b1Local[srcAddr + 0U * blockOffset], params);
        LoadData(b2Local[1U * blockOffset],
            b1Local[srcAddr + 1U * blockOffset], params);
        LoadData(b2Local[2U * blockOffset],
            b1Local[srcAddr + 2U * blockOffset], params);
        LoadData(b2Local[3U * blockOffset],
            b1Local[srcAddr + 3U * blockOffset], params);
    }

    __aicore__ inline void ComputePart(
        const LocalTensor<bfloat16_t>& a1Local,
        const LocalTensor<bfloat16_t>& b1Local,
        const LocalTensor<bfloat16_t>& a2Local,
        const LocalTensor<bfloat16_t>& b2Local,
        const LocalTensor<float>& c1Local,
        uint32_t kPart,
        bool initializeC)
    {
        LoadKPartToL0A(a2Local, a1Local, kPart);
        LoadKPartToL0B(b2Local, b1Local, kPart);

        SetFlag<HardEvent::MTE1_M>(MTE1_TO_M_EVENT);
        WaitFlag<HardEvent::MTE1_M>(MTE1_TO_M_EVENT);

        MmadParams params{};
        params.m = M;
        params.n = N;
        params.k = K_STEP;
        params.cmatrixInitVal = initializeC;
        params.cmatrixSource = false;
        params.unitFlag = 0;
        Mmad(c1Local, a2Local, b2Local, params);
    }

    __aicore__ inline void ComputeGram(
        const LocalTensor<bfloat16_t>& a1Local,
        const LocalTensor<bfloat16_t>& b1Local,
        const LocalTensor<bfloat16_t>& a2Local0,
        const LocalTensor<bfloat16_t>& a2Local1,
        const LocalTensor<bfloat16_t>& a2Local2,
        const LocalTensor<bfloat16_t>& a2Local3,
        const LocalTensor<bfloat16_t>& b2Local0,
        const LocalTensor<bfloat16_t>& b2Local1,
        const LocalTensor<bfloat16_t>& b2Local2,
        const LocalTensor<bfloat16_t>& b2Local3,
        const LocalTensor<float>& c1Local)
    {
        WaitFlag<HardEvent::FIX_M>(CO1_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(SHARED_L1_EVENT);

        ComputePart(
            a1Local, b1Local,
            a2Local0, b2Local0,
            c1Local, 0U, true);
        ComputePart(
            a1Local, b1Local,
            a2Local1, b2Local1,
            c1Local, 1U, false);
        ComputePart(
            a1Local, b1Local,
            a2Local2, b2Local2,
            c1Local, 2U, false);
        ComputePart(
            a1Local, b1Local,
            a2Local3, b2Local3,
            c1Local, 3U, false);

        // Both logical operands have now been read from the shared L1 by
        // PIPE_MTE1. Return ownership once; Cube consumes only L0A/L0B.
        SetFlag<HardEvent::MTE1_MTE2>(SHARED_L1_EVENT);

        // The four physical L0 stages are not reused inside this task. One
        // task-local drain after the final MMAD is sufficient to make every
        // stage writable before the next task starts. The token is consumed
        // immediately, so no M_MTE1 state crosses a task boundary.
        SetFlag<HardEvent::M_MTE1>(TASK_L0_DRAIN_EVENT);
        WaitFlag<HardEvent::M_MTE1>(TASK_L0_DRAIN_EVENT);

        SetFlag<HardEvent::M_FIX>(CO1_EVENT);
        WaitFlag<HardEvent::M_FIX>(CO1_EVENT);
    }

    __aicore__ inline void CommitWorkspace(
        const LocalTensor<float>& c1Local,
        int64_t groupId,
        int32_t slotId)
    {
        const int64_t slotOffset =
            (groupId * SLOT_COUNT + slotId) *
            static_cast<int64_t>(KKT_ELEMS);

        FixpipeParamsV220 params{};
        params.nSize = N;
        params.mSize = M;
        params.srcStride = M;
        params.dstStride = N;
        params.ndNum = 1;
        params.srcNdStride = 0;
        params.dstNdStride = 0;
        Fixpipe(
            kktWorkspaceGm_[slotOffset],
            c1Local,
            params);
    }

private:
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

        // Native VECOUT multi-buffering. A pair is represented by two
        // independent head-sized queues, so each allocated tensor is consumed
        // by exactly one MTE3 DataCopy. Single-head output reuses head0Queue_.
        // Keep four pair batches resident so MTE3 retirement can span two
        // R=8 tasks instead of forcing a drain at every task boundary.
        // The resulting UB footprint remains below the Atlas A2 AIV UB
        // capacity while preserving one tensor per physical output head.
        pipe_->InitBuffer(
            outputHead0Queue_,
            OUTPUT_RETIRE_BATCHES,
            tileElems * sizeof(float));
        pipe_->InitBuffer(
            outputHead1Queue_,
            OUTPUT_RETIRE_BATCHES,
            tileElems * sizeof(float));
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
            BROADCAST_SCRATCH_ELEMS * sizeof(float));

        vToMte2Event_ =
            static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
        mte2ToVEvent_ =
            static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
        outputWindowDrainEvent_ =
            static_cast<event_t>(
                pipe_->FetchEventID(HardEvent::MTE3_S));

        pendingOutputCount_ = 0;
        pendingPairMask_ = 0;
    }

    __aicore__ inline void Process()
    {
        const int64_t aivId =
            static_cast<int64_t>(GetBlockIdx());
        // This MIX kernel is fixed at one AIC plus two AIVs per group.
        // Replace the generic runtime divide/modulo by exact power-of-two
        // mapping. This preserves group/lane semantics while shortening the
        // scalar startup path.
        static_assert(
            AIVS_PER_AIC == 2,
            "AIV group mapping assumes a fixed 1:2 MIX ratio");
        const int64_t groupId = aivId >> 1;
        const int32_t aivLane =
            static_cast<int32_t>(aivId & 1);

        int32_t localRound = 0;
        bool firstBatchPrefetched = false;
        for (int64_t taskId = groupId;
             taskId < tiling_->totalTaskCount;
             taskId += static_cast<int64_t>(tiling_->blockDim),
             ++localRound) {
            const int64_t nextTaskId =
                taskId + static_cast<int64_t>(tiling_->blockDim);
            firstBatchPrefetched = ProcessPipelineTask(
                taskId,
                nextTaskId,
                groupId,
                aivLane,
                localRound,
                firstBatchPrefetched);
        }

        // Retire the final partial output window.
        FlushPendingOutputs();
    }

private:
    __aicore__ inline void FreePendingOutputSlot(int32_t slotId)
    {
        if (slotId == 0) {
            outputHead0Queue_.FreeTensor(pendingHead0Slot0_);
            if ((pendingPairMask_ & 0x1U) != 0U) {
                outputHead1Queue_.FreeTensor(pendingHead1Slot0_);
            }
        } else if (slotId == 1) {
            outputHead0Queue_.FreeTensor(pendingHead0Slot1_);
            if ((pendingPairMask_ & 0x2U) != 0U) {
                outputHead1Queue_.FreeTensor(pendingHead1Slot1_);
            }
        } else if (slotId == 2) {
            outputHead0Queue_.FreeTensor(pendingHead0Slot2_);
            if ((pendingPairMask_ & 0x4U) != 0U) {
                outputHead1Queue_.FreeTensor(pendingHead1Slot2_);
            }
        } else {
            outputHead0Queue_.FreeTensor(pendingHead0Slot3_);
            if ((pendingPairMask_ & 0x8U) != 0U) {
                outputHead1Queue_.FreeTensor(pendingHead1Slot3_);
            }
        }
    }

    __aicore__ inline void FlushPendingOutputs()
    {
        if (pendingOutputCount_ == 0) {
            return;
        }

        // Every pending DataCopy was issued before this event. Keep all four
        // output batches live until PIPE_MTE3 has consumed them, then release
        // the queue tensors in original allocation order.
        SetFlag<HardEvent::MTE3_S>(
            outputWindowDrainEvent_);
        WaitFlag<HardEvent::MTE3_S>(
            outputWindowDrainEvent_);

        for (int32_t slotId = 0;
             slotId < pendingOutputCount_;
             ++slotId) {
            FreePendingOutputSlot(slotId);
        }

        pendingOutputCount_ = 0;
        pendingPairMask_ = 0;
    }

    __aicore__ inline void StorePendingHead0(
        int32_t slotId,
        const LocalTensor<float>& outputHead0Local)
    {
        if (slotId == 0) {
            pendingHead0Slot0_ = outputHead0Local;
        } else if (slotId == 1) {
            pendingHead0Slot1_ = outputHead0Local;
        } else if (slotId == 2) {
            pendingHead0Slot2_ = outputHead0Local;
        } else {
            pendingHead0Slot3_ = outputHead0Local;
        }
    }

    __aicore__ inline void StorePendingHead1(
        int32_t slotId,
        const LocalTensor<float>& outputHead1Local)
    {
        if (slotId == 0) {
            pendingHead1Slot0_ = outputHead1Local;
        } else if (slotId == 1) {
            pendingHead1Slot1_ = outputHead1Local;
        } else if (slotId == 2) {
            pendingHead1Slot2_ = outputHead1Local;
        } else {
            pendingHead1Slot3_ = outputHead1Local;
        }
    }

    __aicore__ inline void RetireSingleOutput(
        const LocalTensor<float>& outputHead0Local)
    {
        StorePendingHead0(
            pendingOutputCount_,
            outputHead0Local);

        ++pendingOutputCount_;
        if (pendingOutputCount_ == OUTPUT_RETIRE_BATCHES) {
            FlushPendingOutputs();
        }
    }

    __aicore__ inline void RetirePairOutput(
        const LocalTensor<float>& outputHead0Local,
        const LocalTensor<float>& outputHead1Local)
    {
        const int32_t slotId = pendingOutputCount_;
        StorePendingHead0(slotId, outputHead0Local);
        StorePendingHead1(slotId, outputHead1Local);
        pendingPairMask_ |= (1U << slotId);

        ++pendingOutputCount_;
        if (pendingOutputCount_ == OUTPUT_RETIRE_BATCHES) {
            FlushPendingOutputs();
        }
    }

    __aicore__ inline void CompleteKktLoadAndReleaseSlot(
        int32_t slotId,
        bool willReuseSlot)
    {
        // The MTE2_V event was published immediately after the KKT copy.
        // Waiting here lets all Factor instructions issued by PIPE_V overlap
        // with the in-flight KKT GM-to-UB transfer.
        WaitFlag<HardEvent::MTE2_V>(
            mte2ToVEvent_);

        // Only KKT completion is ordered before this FREE message. The first
        // prefetch for the following head batch is deliberately issued after
        // this point, so unrelated g/beta MTE2 traffic cannot delay AIC slot
        // reuse.
        if (willReuseSlot) {
            CrossCoreSetFlag<2, PIPE_MTE2>(
                FreeFlagForSlot(slotId));
        }
    }

    __aicore__ inline bool ProcessPipelineTask(
        int64_t taskId,
        int64_t nextTaskId,
        int64_t groupId,
        int32_t aivLane,
        int32_t localRound,
        bool firstBatchPrefetched)
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
                // Exact equivalent of floor(lane * R / 2) and
                // floor((lane + 1) * R / 2), without a per-task DIV.
                // For odd R, lane 1 intentionally retains the extra head.
                const int32_t splitRepeat =
                    tiling_->numRepeat >> 1;
                if (aivLane == 0) {
                    firstRepeat = 0;
                    repeatEnd = splitRepeat;
                } else {
                    firstRepeat = splitRepeat;
                    repeatEnd = tiling_->numRepeat;
                }
            } else {
                // H/Hg==1 fallback: split rows between both AIVs.
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

        // The first g/beta batch is independent of KKT. Start it before the
        // READY wait so its MTE2 can overlap AIC KKT production. It is placed
        // before the KKT copy on the same MTE2 pipeline; therefore DeQue can
        // release PIPE_V to compute Factor while KKT is still transferring.
        if (hasWork && !firstBatchPrefetched) {
            const int64_t firstHead =
                kvHeadId * tiling_->numRepeat;
            const int32_t firstBatchHeads =
                MinI32(
                    GATE_HEAD_BATCH,
                    repeatEnd - firstRepeat);
            if (chunkLen == ROW_ELEMS &&
                rowStart == 0 &&
                rowCount == ROW_ELEMS &&
                firstBatchHeads == GATE_HEAD_BATCH &&
                ((repeatEnd - firstRepeat) & 1) == 0) {
                // Pass compile-time full64/pair constants into the inlined
                // helper. This keeps the general helper but lets the compiler
                // discard its tail and single-head branches on the hot path.
                PrefetchGateAndBetaBatch(
                    firstHead + firstRepeat,
                    start,
                    ROW_ELEMS,
                    GATE_HEAD_BATCH);
            } else {
                PrefetchGateAndBetaBatch(
                    firstHead + firstRepeat,
                    start,
                    chunkLen,
                    firstBatchHeads);
            }
        }

        // Every AIV consumes every READY message, including a lane with no
        // rows or malformed input, so the mode-2 counter remains balanced.
        CrossCoreWaitFlag(
            ReadyFlagForSlot(slotId));

        if (!hasWork) {
            if (willReuseSlot) {
                CrossCoreSetFlag<2, PIPE_MTE2>(
                    FreeFlagForSlot(slotId));
            }
            return false;
        }

        LocalTensor<float> kktLocal =
            BeginLoadKktSlot(
                groupId,
                slotId,
                chunkLen,
                rowStart,
                rowCount);

        return ProcessHeadRange(
            kktLocal,
            kvHeadId,
            start,
            chunkLen,
            rowStart,
            rowCount,
            firstRepeat,
            repeatEnd,
            slotId,
            willReuseSlot,
            nextTaskId,
            aivLane);
    }

    __aicore__ inline bool ProcessHeadRange(
        const LocalTensor<float>& kktLocal,
        int64_t kvHeadId,
        int32_t start,
        int32_t chunkLen,
        int32_t rowStart,
        int32_t rowCount,
        int32_t firstRepeat,
        int32_t repeatEnd,
        int32_t slotId,
        bool willReuseSlot,
        int64_t nextTaskId,
        int32_t aivLane)
    {
        const int64_t firstHead =
            kvHeadId * tiling_->numRepeat;
        const int32_t totalHeads =
            repeatEnd - firstRepeat;
        if (chunkLen == ROW_ELEMS &&
            rowStart == 0 &&
            rowCount == ROW_ELEMS &&
            totalHeads >= GATE_HEAD_BATCH &&
            (totalHeads & 1) == 0) {
            return ProcessFull64HeadRange(
                kktLocal,
                firstHead,
                start,
                firstRepeat,
                repeatEnd,
                slotId,
                willReuseSlot,
                nextTaskId,
                aivLane);
        }

        const int32_t batchCount =
            (totalHeads + GATE_HEAD_BATCH - 1) /
            GATE_HEAD_BATCH;

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

            if (chunkLen != ROW_ELEMS) {
                ZeroTailGateAndBetaBatchSuffix(
                    gBatchLocal,
                    betaBatchLocal,
                    chunkLen,
                    currentBatchHeads);
            }

            if (batchId == 0) {
                // Prepare the complete KKT-independent Factor while the KKT
                // MTE2 copy is in flight. The final elementwise KKT multiply
                // is issued only after CompleteKktLoadAndReleaseSlot().
                if (currentBatchHeads ==
                    GATE_HEAD_BATCH) {
                    LocalTensor<float> outputHead0Local =
                        outputHead0Queue_.AllocTensor<float>();
                    LocalTensor<float> outputHead1Local =
                        outputHead1Queue_.AllocTensor<float>();

                    PrepareHeadPairFactor(
                        gBatchLocal,
                        betaBatchLocal,
                        rowStart,
                        rowCount,
                        outputHead0Local,
                        outputHead1Local);

                    CompleteKktLoadAndReleaseSlot(
                        slotId,
                        willReuseSlot);

                    // Place the next g/beta MTE2 after the slot FREE message,
                    // then overlap it with FinalMul and CopyOut of batch 0.
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

                    FinalizeHeadPairWithKkt(
                        kktLocal,
                        outputHead0Local,
                        outputHead1Local,
                        firstHead + batchRepeat,
                        start,
                        rowStart,
                        rowCount);
                } else {
                    LocalTensor<float> outputLocal =
                        outputHead0Queue_.AllocTensor<float>();

                    PrepareHeadFactor(
                        gBatchLocal,
                        betaBatchLocal,
                        rowStart,
                        rowCount,
                        outputLocal);

                    CompleteKktLoadAndReleaseSlot(
                        slotId,
                        willReuseSlot);

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

                    FinalizeHeadWithKkt(
                        kktLocal,
                        outputLocal,
                        firstHead + batchRepeat,
                        start,
                        rowStart,
                        rowCount);
                }
            } else {
                // KKT is already resident. Preserve the original g/beta
                // ping-pong ordering for all remaining batches.
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
            }

            gQueue_.FreeTensor(
                gBatchLocal);
            betaQueue_.FreeTensor(
                betaBatchLocal);
        }
        return false;
    }

    __aicore__ inline bool ProcessFull64HeadRange(
        const LocalTensor<float>& kktLocal,
        int64_t firstHead,
        int32_t start,
        int32_t firstRepeat,
        int32_t repeatEnd,
        int32_t slotId,
        bool willReuseSlot,
        int64_t nextTaskId,
        int32_t aivLane)
    {
        const int32_t pairCount =
            (repeatEnd - firstRepeat) / GATE_HEAD_BATCH;
        bool nextTaskPrefetched = false;

        // The first pair has already been prefetched before READY. This path
        // is entered only when every local head forms a complete pair. Literal
        // full64 parameters let the inlined helpers fold all row/tail state.
        for (int32_t pairId = 0;
             pairId < pairCount;
             ++pairId) {
            const int32_t pairRepeat =
                firstRepeat + pairId * GATE_HEAD_BATCH;
            const int64_t firstPairHead =
                firstHead + pairRepeat;

            LocalTensor<float> gPairLocal =
                gQueue_.DeQue<float>();
            LocalTensor<bfloat16_t> betaPairLocal =
                betaQueue_.DeQue<bfloat16_t>();

            if (pairId == 0) {
                LocalTensor<float> outputHead0Local =
                    outputHead0Queue_.AllocTensor<float>();
                LocalTensor<float> outputHead1Local =
                    outputHead1Queue_.AllocTensor<float>();

                PrepareHeadPairFactor(
                    gPairLocal,
                    betaPairLocal,
                    0,
                    ROW_ELEMS,
                    outputHead0Local,
                    outputHead1Local);

                CompleteKktLoadAndReleaseSlot(
                    slotId,
                    willReuseSlot);

                if (pairId + 1 < pairCount) {
                    PrefetchGateAndBetaBatch(
                        firstPairHead + GATE_HEAD_BATCH,
                        start,
                        ROW_ELEMS,
                        GATE_HEAD_BATCH);
                } else if (pairCount >= 2) {
                    nextTaskPrefetched =
                        TryPrefetchTaskFirstBatch(
                            nextTaskId,
                            aivLane);
                }

                FinalizeHeadPairWithKkt(
                    kktLocal,
                    outputHead0Local,
                    outputHead1Local,
                    firstPairHead,
                    start,
                    0,
                    ROW_ELEMS);
            } else {
                if (pairId + 1 < pairCount) {
                    PrefetchGateAndBetaBatch(
                        firstPairHead + GATE_HEAD_BATCH,
                        start,
                        ROW_ELEMS,
                        GATE_HEAD_BATCH);
                } else if (pairCount >= 2) {
                    nextTaskPrefetched =
                        TryPrefetchTaskFirstBatch(
                            nextTaskId,
                            aivLane);
                }

                ProcessHeadPairTile(
                    kktLocal,
                    gPairLocal,
                    betaPairLocal,
                    firstPairHead,
                    start,
                    0,
                    ROW_ELEMS);
            }

            gQueue_.FreeTensor(gPairLocal);
            betaQueue_.FreeTensor(betaPairLocal);
        }
        return nextTaskPrefetched;
    }

    __aicore__ inline bool TryPrefetchTaskFirstBatch(
        int64_t taskId,
        int32_t aivLane)
    {
        // Keep this speculative pipeline narrow: only chain stable full64
        // pair tasks with at least four local heads in the current task. Tail,
        // single-head and odd-head tasks retain the original task boundary.
        if (taskId >= tiling_->totalTaskCount ||
            tiling_->numRepeat < AIVS_PER_AIC) {
            return false;
        }

        const int64_t kvHeadId =
            taskId % tiling_->hg;
        const int64_t chunkId =
            taskId / tiling_->hg;
        const int32_t start =
            chunkOffsetsGm_.GetValue(chunkId);
        const int32_t end =
            chunkOffsetsGm_.GetValue(chunkId + 1);
        if (start < 0 || end > tiling_->t ||
            end - start != ROW_ELEMS) {
            return false;
        }

        const int32_t splitRepeat =
            tiling_->numRepeat >> 1;
        const int32_t firstRepeat =
            aivLane == 0 ? 0 : splitRepeat;
        const int32_t repeatEnd =
            aivLane == 0 ? splitRepeat : tiling_->numRepeat;
        const int32_t localHeads =
            repeatEnd - firstRepeat;
        if (localHeads < GATE_HEAD_BATCH ||
            (localHeads & 1) != 0) {
            return false;
        }

        const int64_t firstHead =
            kvHeadId * tiling_->numRepeat + firstRepeat;
        PrefetchGateAndBetaBatch(
            firstHead,
            start,
            ROW_ELEMS,
            GATE_HEAD_BATCH);
        return true;
    }

    __aicore__ inline LocalTensor<float> BeginLoadKktSlot(
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
            
        }

        // kktBuf_ is a reusable TBuf. Ensure the previous Vector reads have
        // completed before MTE2 overwrites the same UB storage.
        SetFlag<HardEvent::V_MTE2>(
            vToMte2Event_);
        WaitFlag<HardEvent::V_MTE2>(
            vToMte2Event_);

        const int64_t slotOffset =
            (groupId * SLOT_COUNT + slotId) *
            static_cast<int64_t>(KKT_ELEMS);
        GlobalTensor<float> taskKkt =
            kktWorkspaceGm_[slotOffset];
        if (chunkLen == ROW_ELEMS) {
            DataCopy(
                kktLocal,
                taskKkt[rowStart * ROW_ELEMS],
                static_cast<uint32_t>(
                    rowCount * ROW_ELEMS));
        } else {
            const int32_t alignedRowElems =
                ((chunkLen + 7) / 8) * 8;
            const uint8_t rightPad =
                static_cast<uint8_t>(
                    alignedRowElems - chunkLen);
            const uint32_t dstStrideBlocks =
                static_cast<uint32_t>(
                    (ROW_ELEMS - alignedRowElems) / 8);
            const uint32_t srcStrideBytes =
                static_cast<uint32_t>(
                    (ROW_ELEMS - chunkLen) *
                    sizeof(float));
            DataCopyExtParams copyParams{
                static_cast<uint16_t>(rowCount),
                static_cast<uint32_t>(
                    chunkLen * sizeof(float)),
                srcStrideBytes,
                dstStrideBlocks,
                0};
            DataCopyPadExtParams<float> padParams{
                true, 0, rightPad, 0.0f};
            constexpr int32_t sourceRowStride = ROW_ELEMS;
            DataCopyPad(
                kktLocal,
                taskKkt[rowStart * sourceRowStride],
                copyParams,
                padParams);
        }

        // Publish completion to PIPE_V but intentionally do not wait here.
        // The first batch computes Factor before consuming this event.
        SetFlag<HardEvent::MTE2_V>(
            mte2ToVEvent_);
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


    __aicore__ inline void PrepareHeadPairFactor(
        const LocalTensor<float>& gPairLocal,
        const LocalTensor<bfloat16_t>& betaPairLocal,
        int32_t rowStart,
        int32_t rowCount,
        LocalTensor<float> outputHead0Local,
        LocalTensor<float> outputHead1Local)
    {
        LocalTensor<float> negGPairLocal =
            negGPairBuf_.Get<float>();
        LocalTensor<float> scratch =
            broadcastScratchBuf_.Get<float>();
        LocalTensor<uint8_t> negativeMaskPair =
            negativeMaskPairBuf_.Get<uint8_t>();

        const int32_t activeElems =
            rowCount * ROW_ELEMS;
        constexpr int32_t maskHeadStrideBytes =
            (KKT_ELEMS + 7) / 8;
        LocalTensor<uint8_t> negativeMaskHead0 =
            negativeMaskPair;
        LocalTensor<uint8_t> negativeMaskHead1 =
            negativeMaskPair[maskHeadStrideBytes];

        // Brcb requires a 32-byte-aligned source address.
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

        Muls(
            negGPairLocal,
            gPairLocal,
            -1.0f,
            GATE_HEAD_BATCH * ROW_ELEMS);
        

        if (rowStart == 0 &&
            rowCount == ROW_ELEMS) {
            Brcb(
                scratch,
                gPairLocal,
                static_cast<uint8_t>(
                    GATE_HEAD_BATCH *
                    ROW_ELEMS / 8),
                {1, 8});
        } else {
            Brcb(
                scratch,
                gPairLocal[alignedRowStart],
                rowBrcbRepeats,
                {1, 8});
            Brcb(
                scratch[scratchHeadStride],
                gPairLocal[
                    ROW_ELEMS + alignedRowStart],
                rowBrcbRepeats,
                {1, 8});
        }
        PipeBarrier<PIPE_V>();

        Add(
            outputHead0Local,
            negGPairLocal,
            scratch[scratchRowOffset],
            static_cast<uint64_t>(ROW_ELEMS),
            static_cast<uint8_t>(rowCount),
            {1, 1, 0, 8, 0, 1});
        Add(
            outputHead1Local,
            negGPairLocal[ROW_ELEMS],
            scratch[
                scratchHeadStride +
                scratchRowOffset],
            static_cast<uint64_t>(ROW_ELEMS),
            static_cast<uint8_t>(rowCount),
            {1, 1, 0, 8, 0, 1});
        PipeBarrier<PIPE_V>();

        CompareScalar(
            negativeMaskHead0,
            outputHead0Local,
            0.0f,
            CMPMODE::LT,
            activeElems);
        CompareScalar(
            negativeMaskHead1,
            outputHead1Local,
            0.0f,
            CMPMODE::LT,
            activeElems);
        PipeBarrier<PIPE_V>();

        Select(
            outputHead0Local,
            negativeMaskHead0,
            outputHead0Local,
            NEGATIVE_INF_PROXY,
            SELMODE::VSEL_TENSOR_SCALAR_MODE,
            activeElems);
        Select(
            outputHead1Local,
            negativeMaskHead1,
            outputHead1Local,
            NEGATIVE_INF_PROXY,
            SELMODE::VSEL_TENSOR_SCALAR_MODE,
            activeElems);
        PipeBarrier<PIPE_V>();

        Exp(
            outputHead0Local,
            outputHead0Local,
            activeElems);
        Exp(
            outputHead1Local,
            outputHead1Local,
            activeElems);
        PipeBarrier<PIPE_V>();

        // Reuse negGPairLocal as compact FP32 beta storage.
        Cast(
            negGPairLocal,
            betaPairLocal,
            RoundMode::CAST_NONE,
            GATE_HEAD_BATCH * ROW_ELEMS);
        PipeBarrier<PIPE_V>();

        if (rowStart == 0 &&
            rowCount == ROW_ELEMS) {
            Brcb(
                scratch,
                negGPairLocal,
                static_cast<uint8_t>(
                    GATE_HEAD_BATCH *
                    ROW_ELEMS / 8),
                {1, 8});
        } else {
            Brcb(
                scratch,
                negGPairLocal[alignedRowStart],
                rowBrcbRepeats,
                {1, 8});
            Brcb(
                scratch[scratchHeadStride],
                negGPairLocal[
                    ROW_ELEMS + alignedRowStart],
                rowBrcbRepeats,
                {1, 8});
        }
        PipeBarrier<PIPE_V>();

        Mul(
            outputHead0Local,
            outputHead0Local,
            scratch[scratchRowOffset],
            static_cast<uint64_t>(ROW_ELEMS),
            static_cast<uint8_t>(rowCount),
            {1, 1, 0, 8, 8, 1});
        Mul(
            outputHead1Local,
            outputHead1Local,
            scratch[
                scratchHeadStride +
                scratchRowOffset],
            static_cast<uint64_t>(ROW_ELEMS),
            static_cast<uint8_t>(rowCount),
            {1, 1, 0, 8, 8, 1});
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void FinalizeHeadPairWithKkt(
        const LocalTensor<float>& kktLocal,
        LocalTensor<float> outputHead0Local,
        LocalTensor<float> outputHead1Local,
        int64_t firstHeadId,
        int32_t start,
        int32_t rowStart,
        int32_t rowCount)
    {
        const int32_t activeElems =
            rowCount * ROW_ELEMS;

        Mul(
            outputHead0Local,
            kktLocal,
            outputHead0Local,
            activeElems);
        Mul(
            outputHead1Local,
            kktLocal,
            outputHead1Local,
            activeElems);
        

        outputHead0Queue_.EnQue(
            outputHead0Local);
        outputHead1Queue_.EnQue(
            outputHead1Local);
        CopyOutHeadPair(
            firstHeadId,
            start,
            rowStart,
            rowCount,
            activeElems);
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
        LocalTensor<float> outputHead0Local =
            outputHead0Queue_.AllocTensor<float>();
        LocalTensor<float> outputHead1Local =
            outputHead1Queue_.AllocTensor<float>();

        PrepareHeadPairFactor(
            gPairLocal,
            betaPairLocal,
            rowStart,
            rowCount,
            outputHead0Local,
            outputHead1Local);
        FinalizeHeadPairWithKkt(
            kktLocal,
            outputHead0Local,
            outputHead1Local,
            firstHeadId,
            start,
            rowStart,
            rowCount);
    }

    __aicore__ inline void PrepareHeadFactor(
        const LocalTensor<float>& gLocal,
        const LocalTensor<bfloat16_t>& betaLocal,
        int32_t rowStart,
        int32_t rowCount,
        LocalTensor<float> outputLocal)
    {
        LocalTensor<float> negGLocal =
            negGSingleBuf_.Get<float>();
        LocalTensor<float> scratch =
            broadcastScratchBuf_.Get<float>();
        LocalTensor<uint8_t> negativeMask =
            negativeMaskSingleBuf_.Get<uint8_t>();

        const int32_t activeElems =
            rowCount * ROW_ELEMS;

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
    }

    __aicore__ inline void FinalizeHeadWithKkt(
        const LocalTensor<float>& kktLocal,
        LocalTensor<float> outputLocal,
        int64_t headId,
        int32_t start,
        int32_t rowStart,
        int32_t rowCount)
    {
        const int32_t activeElems =
            rowCount * ROW_ELEMS;

        Mul(
            outputLocal,
            kktLocal,
            outputLocal,
            activeElems);
      

        outputHead0Queue_.EnQue(
            outputLocal);
        CopyOutTile(
            headId,
            start,
            rowStart,
            rowCount);
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
        LocalTensor<float> outputLocal =
            outputHead0Queue_.AllocTensor<float>();

        PrepareHeadFactor(
            gLocal,
            betaLocal,
            rowStart,
            rowCount,
            outputLocal);
        FinalizeHeadWithKkt(
            kktLocal,
            outputLocal,
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
        LocalTensor<float> outputHead0Local =
            outputHead0Queue_.DeQue<float>();
        LocalTensor<float> outputHead1Local =
            outputHead1Queue_.DeQue<float>();

        const int64_t firstOutputOffset =
            (firstHeadId * tiling_->t + start + rowStart) *
            static_cast<int64_t>(ROW_ELEMS);
        const int64_t secondOutputOffset =
            ((firstHeadId + 1) * tiling_->t + start + rowStart) *
            static_cast<int64_t>(ROW_ELEMS);

        DataCopy(
            outputGm_[firstOutputOffset],
            outputHead0Local,
            static_cast<uint32_t>(activeElems));
        DataCopy(
            outputGm_[secondOutputOffset],
            outputHead1Local,
            static_cast<uint32_t>(activeElems));

        // Keep both physical tensors alive while MTE3 consumes them. They
        // are released together with the next output batch, even when that
        // next batch belongs to a different logical task.
        RetirePairOutput(
            outputHead0Local,
            outputHead1Local);
    }


    __aicore__ inline void CopyOutTile(
        int64_t headId,
        int32_t start,
        int32_t rowStart,
        int32_t rowCount)
    {
        LocalTensor<float> outputLocal =
            outputHead0Queue_.DeQue<float>();
        const int64_t outputOffset =
            (headId * tiling_->t + start + rowStart) *
            static_cast<int64_t>(ROW_ELEMS);

        DataCopy(
            outputGm_[outputOffset],
            outputLocal,
            static_cast<uint32_t>(
                rowCount * ROW_ELEMS));

        RetireSingleOutput(outputLocal);
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
    TQue<TPosition::VECOUT, 1> outputHead0Queue_;
    TQue<TPosition::VECOUT, 1> outputHead1Queue_;
    TQue<TPosition::VECIN, 1> gQueue_;
    TQue<TPosition::VECIN, 1> betaQueue_;
    TBuf<TPosition::VECCALC> negGSingleBuf_;
    TBuf<TPosition::VECCALC> negGPairBuf_;
    TBuf<TPosition::VECCALC> negativeMaskSingleBuf_;
    TBuf<TPosition::VECCALC> negativeMaskPairBuf_;
    TBuf<TPosition::VECCALC> broadcastScratchBuf_;

    event_t vToMte2Event_;
    event_t mte2ToVEvent_;
    event_t outputWindowDrainEvent_;

    // Four-batch output window. Slot identity follows allocation order, not
    // logical-task identity, so the window intentionally spans task
    // boundaries and amortizes one MTE3 drain across more output copies.
    LocalTensor<float> pendingHead0Slot0_;
    LocalTensor<float> pendingHead0Slot1_;
    LocalTensor<float> pendingHead0Slot2_;
    LocalTensor<float> pendingHead0Slot3_;
    LocalTensor<float> pendingHead1Slot0_;
    LocalTensor<float> pendingHead1Slot1_;
    LocalTensor<float> pendingHead1Slot2_;
    LocalTensor<float> pendingHead1Slot3_;
    int32_t pendingOutputCount_ = 0;
    uint32_t pendingPairMask_ = 0;
};


class PackedGramMmadCubeStageK128L1PingPong {
public:
    static constexpr uint32_t K_DIM = 128U;

    static constexpr uint32_t M = 64;
    static constexpr uint32_t N = 64;
    static constexpr uint32_t K_STEP = 64;
    static constexpr uint32_t K_PARTS = K_DIM / K_STEP;
    static constexpr uint32_t FRACTAL_M = 16;
    static constexpr uint32_t FRACTAL_K = 16;
    static constexpr uint32_t N_BLOCKS = N / FRACTAL_M;

    static constexpr uint32_t A1_ELEMS = M * K_DIM;
    static constexpr uint32_t B1_ELEMS = N * K_DIM;
    static constexpr uint32_t A2_ELEMS = M * K_STEP;
    static constexpr uint32_t B2_ELEMS = K_STEP * N;
    static constexpr uint32_t CO1_ELEMS = M * N;

    static constexpr uint32_t A1_BYTES =
        A1_ELEMS * sizeof(bfloat16_t);
    static constexpr uint32_t B1_BYTES =
        B1_ELEMS * sizeof(bfloat16_t);

    // A and B alias one packed K tile per task stage. K=128 therefore uses
    // 16 KiB per stage and 32 KiB for the two-stage task ping-pong.
    static constexpr uint32_t SHARED_L1_ADDR0 = 0U;
    static constexpr uint32_t SHARED_L1_ADDR1 =
        SHARED_L1_ADDR0 + A1_BYTES;
    static constexpr uint32_t L1_USED_BYTES =
        SHARED_L1_ADDR1 + A1_BYTES;
    static constexpr uint32_t L1_CAPACITY_BYTES = 512U * 1024U;

    static constexpr uint32_t A2_BYTES =
        A2_ELEMS * sizeof(bfloat16_t);
    static constexpr uint32_t B2_BYTES =
        B2_ELEMS * sizeof(bfloat16_t);
    static constexpr uint32_t L0A_CAPACITY_BYTES = 64U * 1024U;
    static constexpr uint32_t L0B_CAPACITY_BYTES = 64U * 1024U;
    static constexpr uint32_t A2_STAGE_BYTES =
        L0A_CAPACITY_BYTES / 2U;
    static constexpr uint32_t B2_STAGE_BYTES =
        L0B_CAPACITY_BYTES / 2U;
    static constexpr uint32_t A2_ADDR0 = 0;
    static constexpr uint32_t A2_ADDR1 = A2_STAGE_BYTES;
    static constexpr uint32_t B2_ADDR0 = 0;
    static constexpr uint32_t B2_ADDR1 = B2_STAGE_BYTES;
    static constexpr uint32_t CO1_ADDR = 0;

    // One ownership token per physical shared-L1 task stage. A and B loads
    // are ordered on PIPE_MTE1, so the token is returned after the final B read.
    static constexpr event_t SHARED_L1_EVENT0 = EVENT_ID0;
    static constexpr event_t SHARED_L1_EVENT1 = EVENT_ID1;

    static constexpr event_t L0_A_EVENT0 = EVENT_ID0;
    static constexpr event_t L0_A_EVENT1 = EVENT_ID1;
    static constexpr event_t L0_B_EVENT0 = EVENT_ID2;
    static constexpr event_t L0_B_EVENT1 = EVENT_ID3;
    static constexpr event_t MTE1_TO_M_EVENT = EVENT_ID0;
    static constexpr event_t CO1_EVENT = EVENT_ID0;

    static_assert(K_PARTS == 2U,
        "K=128 must contain exactly two 64-wide K parts");
    static_assert(A1_ELEMS == B1_ELEMS,
        "A1/B1 aliases require identical packed tile sizes");
    static_assert((SHARED_L1_ADDR0 % 32U) == 0U &&
                  (SHARED_L1_ADDR1 % 32U) == 0U,
        "Shared L1 stage addresses must be 32-byte aligned");
    static_assert(A1_BYTES % 32U == 0U,
        "A1 task stage must be 32-byte aligned");
    static_assert(B1_BYTES % 32U == 0U,
        "B1 task stage must be 32-byte aligned");
    static_assert(L1_USED_BYTES <= L1_CAPACITY_BYTES,
        "Two A1/B1 task stages exceed Atlas A2 L1 capacity");
    static_assert(A2_BYTES % 512U == 0U,
        "A2 tile must be 512-byte aligned");
    static_assert(B2_BYTES % 512U == 0U,
        "B2 tile must be 512-byte aligned");
    static_assert(A2_BYTES <= A2_STAGE_BYTES,
        "A2 tile exceeds one L0A stage");
    static_assert(B2_BYTES <= B2_STAGE_BYTES,
        "B2 tile exceeds one L0B stage");
    static_assert((CO1_ELEMS * sizeof(float)) % 1024U == 0U,
        "CO1 tile must be 1024-byte aligned");

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
        InitSocState();

        const int64_t groupId =
            static_cast<int64_t>(GetBlockIdx());
        const int64_t taskStride =
            static_cast<int64_t>(tiling_->blockDim);

        // Each pair is two typed views of the same physical L1 stage.
        LocalTensor<bfloat16_t> a1Local0(
            TPosition::A1,
            SHARED_L1_ADDR0,
            A1_ELEMS);
        LocalTensor<bfloat16_t> b1Local0(
            TPosition::B1,
            SHARED_L1_ADDR0,
            B1_ELEMS);
        LocalTensor<bfloat16_t> a1Local1(
            TPosition::A1,
            SHARED_L1_ADDR1,
            A1_ELEMS);
        LocalTensor<bfloat16_t> b1Local1(
            TPosition::B1,
            SHARED_L1_ADDR1,
            B1_ELEMS);

        LocalTensor<bfloat16_t> a2Local0(
            TPosition::A2,
            A2_ADDR0,
            A2_ELEMS);
        LocalTensor<bfloat16_t> a2Local1(
            TPosition::A2,
            A2_ADDR1,
            A2_ELEMS);
        LocalTensor<bfloat16_t> b2Local0(
            TPosition::B2,
            B2_ADDR0,
            B2_ELEMS);
        LocalTensor<bfloat16_t> b2Local1(
            TPosition::B2,
            B2_ADDR1,
            B2_ELEMS);
        LocalTensor<float> c1Local(
            TPosition::CO1,
            CO1_ADDR,
            CO1_ELEMS);

        InitPersistentEvents();

        // Warm both task stages. MTE2 executes these copies in order; MTE1
        // may start consuming stage 0 as soon as its own A/B copies complete,
        // while MTE2 continues filling stage 1.
        const int64_t firstTask = groupId;
        if (firstTask < tiling_->totalTaskCount) {
            PrefetchTaskToL1<0>(
                firstTask,
                a1Local0);
        }
        const int64_t secondTask = firstTask + taskStride;
        if (secondTask < tiling_->totalTaskCount) {
            PrefetchTaskToL1<1>(
                secondTask,
                a1Local1);
        }

        int32_t localRound = 0;
        for (int64_t taskId = groupId;
             taskId < tiling_->totalTaskCount;
             taskId += taskStride,
             ++localRound) {
            const int32_t l1Stage = localRound & 1;
            const int32_t slotId =
                localRound & (SLOT_COUNT - 1);

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
                IsValidChunk(start, end, chunkLen);

            // Preserve the validated invariant: no M_MTE1 ownership token is
            // allowed to cross a task boundary.
            InitL0Events();

            if (validChunk) {
                if (l1Stage == 0) {
                    ComputeGram<0>(
                        a1Local0,
                        b1Local0,
                        a2Local0,
                        a2Local1,
                        b2Local0,
                        b2Local1,
                        c1Local);
                } else {
                    ComputeGram<1>(
                        a1Local1,
                        b1Local1,
                        a2Local0,
                        a2Local1,
                        b2Local0,
                        b2Local1,
                        c1Local);
                }
            }

            // Refill the stage just released by MTE1 with task r+2. This is
            // issued before Fixpipe/slot retirement and before the next loop's
            // Cube work, creating the steady-state MTE2-vs-Cube overlap.
            const int64_t prefetchTask =
                taskId + 2 * taskStride;
            if (prefetchTask < tiling_->totalTaskCount) {
                if (l1Stage == 0) {
                    PrefetchTaskToL1<0>(
                        prefetchTask,
                        a1Local0);
                } else {
                    PrefetchTaskToL1<1>(
                        prefetchTask,
                        a1Local1);
                }
            }

            // Keep workspace retirement late, exactly as in the validated
            // direct-MMAD path. The wait protects only GM slot overwrite.
            if (localRound >= SLOT_COUNT) {
                CrossCoreWaitFlag(
                    FreeFlagForSlot(slotId));
            }

            if (validChunk) {
                CommitWorkspace(
                    c1Local,
                    groupId,
                    slotId);
            }

            CrossCoreSetFlag<2, PIPE_FIX>(
                ReadyFlagForSlot(slotId));

            if (validChunk) {
                SetFlag<HardEvent::FIX_M>(CO1_EVENT);
            }

            DrainL0Events();
        }

        DrainPersistentEvents();
    }

private:
    __aicore__ inline bool IsValidChunk(
        int32_t start,
        int32_t end,
        int32_t chunkLen) const
    {
        return start >= 0 &&
            end <= tiling_->t &&
            chunkLen > 0 &&
            chunkLen <= static_cast<int32_t>(M);
    }

    __aicore__ inline void InitPersistentEvents()
    {
        SetFlag<HardEvent::MTE1_MTE2>(SHARED_L1_EVENT0);
        SetFlag<HardEvent::MTE1_MTE2>(SHARED_L1_EVENT1);
        SetFlag<HardEvent::FIX_M>(CO1_EVENT);
    }

    __aicore__ inline void InitL0Events()
    {
        SetFlag<HardEvent::M_MTE1>(L0_A_EVENT0);
        SetFlag<HardEvent::M_MTE1>(L0_A_EVENT1);
        SetFlag<HardEvent::M_MTE1>(L0_B_EVENT0);
        SetFlag<HardEvent::M_MTE1>(L0_B_EVENT1);
    }

    __aicore__ inline void DrainL0Events()
    {
        WaitFlag<HardEvent::M_MTE1>(L0_A_EVENT0);
        WaitFlag<HardEvent::M_MTE1>(L0_A_EVENT1);
        WaitFlag<HardEvent::M_MTE1>(L0_B_EVENT0);
        WaitFlag<HardEvent::M_MTE1>(L0_B_EVENT1);
    }

    __aicore__ inline void DrainPersistentEvents()
    {
        WaitFlag<HardEvent::MTE1_MTE2>(SHARED_L1_EVENT0);
        WaitFlag<HardEvent::MTE1_MTE2>(SHARED_L1_EVENT1);
        WaitFlag<HardEvent::FIX_M>(CO1_EVENT);

        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ClearSharedL1(
        const LocalTensor<bfloat16_t>& sharedL1Local)
    {
        constexpr uint16_t sharedBlocks =
            static_cast<uint16_t>(A1_BYTES / 32U);

        InitConstValue(
            sharedL1Local,
            {1, sharedBlocks, 0, ToBfloat16(0.0f)});
    }

    __aicore__ inline void CopyOnePackedOperand(
        const LocalTensor<bfloat16_t>& dstLocal,
        const GlobalTensor<bfloat16_t>& srcGlobal,
        int32_t chunkLen)
    {
        Nd2NzParams copyParams{};
        copyParams.ndNum = 1;
        copyParams.nValue =
            static_cast<uint16_t>(chunkLen);
        copyParams.dValue =
            static_cast<uint16_t>(K_DIM);
        copyParams.srcNdMatrixStride = 0;
        copyParams.srcDValue =
            static_cast<uint16_t>(tiling_->hg * K_DIM);
        copyParams.dstNzC0Stride =
            static_cast<uint16_t>(M);
        copyParams.dstNzNStride = 1;
        copyParams.dstNzMatrixStride = 0;
        DataCopy(
            dstLocal,
            srcGlobal,
            copyParams);
    }

    template <uint32_t L1_STAGE>
    __aicore__ inline void PrefetchTaskToL1(
        int64_t taskId,
        const LocalTensor<bfloat16_t>& sharedL1Local)
    {
        static_assert(L1_STAGE < 2U, "Invalid L1 stage");

        const int64_t kvHeadId =
            taskId % tiling_->hg;
        const int64_t chunkId =
            taskId / tiling_->hg;
        const int32_t start =
            chunkOffsetsGm_.GetValue(chunkId);
        const int32_t end =
            chunkOffsetsGm_.GetValue(chunkId + 1);
        const int32_t chunkLen = end - start;

        // Invalid tasks deliberately consume no L1 token. The stage keeps its
        // writable MTE1_MTE2 token and the corresponding compute is skipped.
        if (!IsValidChunk(start, end, chunkLen)) {
            return;
        }

        constexpr event_t SHARED_EVENT =
            L1_STAGE == 0U ? SHARED_L1_EVENT0 : SHARED_L1_EVENT1;

        WaitFlag<HardEvent::MTE1_MTE2>(SHARED_EVENT);

        if (chunkLen < static_cast<int32_t>(M)) {
            ClearSharedL1(sharedL1Local);
        }

        const int64_t kOffset =
            (static_cast<int64_t>(start) *
                tiling_->hg + kvHeadId) * K_DIM;

        // One GM->L1 transfer feeds both typed A1/B1 views.
        CopyOnePackedOperand(
            sharedL1Local,
            kGm_[kOffset],
            chunkLen);
        SetFlag<HardEvent::MTE2_MTE1>(SHARED_EVENT);
    }

    __aicore__ inline void LoadKPartToL0A(
        const LocalTensor<bfloat16_t>& a2Local,
        const LocalTensor<bfloat16_t>& a1Local,
        uint32_t kPart)
    {
        const uint32_t srcAddr =
            kPart * K_STEP * M;

        LoadData3DParamsV2<bfloat16_t> params{};
        params.l1H = 1;
        params.l1W = M;
        params.channelSize = K_STEP;
        params.kExtension = K_STEP;
        params.mExtension = M;
        params.mStartPt = 0;
        params.kStartPt = 0;
        LoadData(
            a2Local,
            a1Local[srcAddr],
            params);
    }

    __aicore__ inline void LoadKPartToL0B(
        const LocalTensor<bfloat16_t>& b2Local,
        const LocalTensor<bfloat16_t>& b1Local,
        uint32_t kPart)
    {
        const uint32_t srcAddr =
            kPart * K_STEP * N;

        LoadData2DParams params{};
        params.repeatTimes =
            static_cast<uint8_t>(N_BLOCKS);
        params.srcStride = 1;
        params.dstGap = 0;
        params.ifTranspose = false;

        constexpr uint32_t dstOffset =
            N * FRACTAL_K;
        constexpr uint32_t srcOffset =
            N * FRACTAL_K;

        LoadData(b2Local[0U * dstOffset],
            b1Local[srcAddr + 0U * srcOffset], params);
        LoadData(b2Local[1U * dstOffset],
            b1Local[srcAddr + 1U * srcOffset], params);
        LoadData(b2Local[2U * dstOffset],
            b1Local[srcAddr + 2U * srcOffset], params);
        LoadData(b2Local[3U * dstOffset],
            b1Local[srcAddr + 3U * srcOffset], params);
    }

    template <uint32_t K_PART, uint32_t L1_STAGE>
    __aicore__ inline void ComputeOneKPart(
        const LocalTensor<bfloat16_t>& a1Local,
        const LocalTensor<bfloat16_t>& b1Local,
        const LocalTensor<bfloat16_t>& a2Local0,
        const LocalTensor<bfloat16_t>& a2Local1,
        const LocalTensor<bfloat16_t>& b2Local0,
        const LocalTensor<bfloat16_t>& b2Local1,
        const LocalTensor<float>& c1Local)
    {
        static_assert(K_PART < K_PARTS, "Invalid K-part index");
        static_assert(L1_STAGE < 2U, "Invalid L1 stage");

        constexpr bool USE_L0_STAGE1 =
            ((K_PART & 1U) != 0U);
        constexpr event_t A_L0_EVENT =
            USE_L0_STAGE1 ? L0_A_EVENT1 : L0_A_EVENT0;
        constexpr event_t B_L0_EVENT =
            USE_L0_STAGE1 ? L0_B_EVENT1 : L0_B_EVENT0;
        constexpr event_t SHARED_L1_EVENT =
            L1_STAGE == 0U ? SHARED_L1_EVENT0 : SHARED_L1_EVENT1;

        WaitFlag<HardEvent::M_MTE1>(A_L0_EVENT);
        if constexpr (K_PART == 0U) {
            WaitFlag<HardEvent::MTE2_MTE1>(SHARED_L1_EVENT);
        }
        if constexpr (USE_L0_STAGE1) {
            LoadKPartToL0A(a2Local1, a1Local, K_PART);
        } else {
            LoadKPartToL0A(a2Local0, a1Local, K_PART);
        }
        WaitFlag<HardEvent::M_MTE1>(B_L0_EVENT);
        if constexpr (USE_L0_STAGE1) {
            LoadKPartToL0B(b2Local1, b1Local, K_PART);
        } else {
            LoadKPartToL0B(b2Local0, b1Local, K_PART);
        }
        if constexpr (K_PART + 1U == K_PARTS) {
            // The final B read is the last PIPE_MTE1 consumer of this stage.
            SetFlag<HardEvent::MTE1_MTE2>(SHARED_L1_EVENT);
        }

        SetFlag<HardEvent::MTE1_M>(MTE1_TO_M_EVENT);
        WaitFlag<HardEvent::MTE1_M>(MTE1_TO_M_EVENT);

        MmadParams params{};
        params.m = M;
        params.n = N;
        params.k = K_STEP;
        params.cmatrixInitVal = (K_PART == 0U);
        params.cmatrixSource = false;
        params.unitFlag = 0;
        if constexpr (USE_L0_STAGE1) {
            Mmad(c1Local, a2Local1, b2Local1, params);
        } else {
            Mmad(c1Local, a2Local0, b2Local0, params);
        }

        SetFlag<HardEvent::M_MTE1>(A_L0_EVENT);
        SetFlag<HardEvent::M_MTE1>(B_L0_EVENT);
    }

    template <uint32_t L1_STAGE>
    __aicore__ inline void ComputeGram(
        const LocalTensor<bfloat16_t>& a1Local,
        const LocalTensor<bfloat16_t>& b1Local,
        const LocalTensor<bfloat16_t>& a2Local0,
        const LocalTensor<bfloat16_t>& a2Local1,
        const LocalTensor<bfloat16_t>& b2Local0,
        const LocalTensor<bfloat16_t>& b2Local1,
        const LocalTensor<float>& c1Local)
    {
        static_assert(L1_STAGE < 2U, "Invalid L1 stage");

        WaitFlag<HardEvent::FIX_M>(CO1_EVENT);

        ComputeOneKPart<0, L1_STAGE>(
            a1Local,
            b1Local,
            a2Local0,
            a2Local1,
            b2Local0,
            b2Local1,
            c1Local);
        ComputeOneKPart<1, L1_STAGE>(
            a1Local,
            b1Local,
            a2Local0,
            a2Local1,
            b2Local0,
            b2Local1,
            c1Local);

        SetFlag<HardEvent::M_FIX>(CO1_EVENT);
        WaitFlag<HardEvent::M_FIX>(CO1_EVENT);
    }

    __aicore__ inline void CommitWorkspace(
        const LocalTensor<float>& c1Local,
        int64_t groupId,
        int32_t slotId)
    {
        const int64_t slotOffset =
            (groupId * SLOT_COUNT + slotId) *
            static_cast<int64_t>(KKT_ELEMS);

        FixpipeParamsV220 params{};
        params.nSize = N;
        params.mSize = M;
        params.srcStride = M;
        params.dstStride = N;
        params.ndNum = 1;
        params.srcNdStride = 0;
        params.dstNdStride = 0;
        Fixpipe(
            kktWorkspaceGm_[slotOffset],
            c1Local,
            params);
    }

private:
    const ChunkScaledDotKktTilingData* tiling_ = nullptr;
    GlobalTensor<bfloat16_t> kGm_;
    GlobalTensor<int32_t> chunkOffsetsGm_;
    GlobalTensor<float> kktWorkspaceGm_;
};

} // namespace NsChunkScaledDotKkt

#endif // CHUNK_SCALED_DOT_KKT_H