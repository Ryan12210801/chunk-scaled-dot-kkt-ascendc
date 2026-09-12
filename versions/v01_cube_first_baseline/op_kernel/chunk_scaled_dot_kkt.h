/*!
 * \file chunk_scaled_dot_kkt.h
 * \brief Cube-first V1 kernel for ChunkScaledDotKkt.
 */

#ifndef CHUNK_SCALED_DOT_KKT_H
#define CHUNK_SCALED_DOT_KKT_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "chunk_scaled_dot_kkt_tiling_data.h"

namespace NsChunkScaledDotKkt {

using namespace AscendC;

constexpr int32_t BT = 64;
constexpr int32_t ROW_ELEMS = 64;
constexpr int32_t KKT_ELEMS = BT * BT;
constexpr int32_t MASK_BYTES = 32;
constexpr float NEGATIVE_INF_PROXY = -3.402823466e+38F;

template <uint32_t K_DIM>
class ChunkScaledDotKkt {
public:
    static constexpr uint32_t K_ELEMS = BT * K_DIM;

    using AType = MatmulType<TPosition::VECOUT, CubeFormat::ND, bfloat16_t, false>;
    using BType = MatmulType<TPosition::VECOUT, CubeFormat::ND, bfloat16_t, true>;
    using CType = MatmulType<TPosition::VECIN, CubeFormat::ND, float, false>;
    using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float, false>;
    using MatmulObj = Matmul<AType, BType, CType, BiasType>;

    __aicore__ inline ChunkScaledDotKkt(MatmulObj& mm, TPipe* pipe)
        : mm_(mm), pipe_(pipe)
    {
    }

    __aicore__ inline void Init(
        GM_ADDR k,
        GM_ADDR beta,
        GM_ADDR gCumsum,
        GM_ADDR chunkOffsets,
        GM_ADDR output,
        const ChunkScaledDotKktTilingData* tilingData)
    {
        tiling_ = tilingData;

        kGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(k),
            tiling_->t * tiling_->hg * K_DIM);
        betaGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(beta),
            tiling_->h * tiling_->t);
        gGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(gCumsum),
            tiling_->h * tiling_->t);
        chunkOffsetsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(chunkOffsets),
            tiling_->numChunks + 1);
        outputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(output),
            tiling_->h * tiling_->t * BT);

        // K and KKT are shared by all attention heads mapped to one KV head.
        pipe_->InitBuffer(kBuf_, K_ELEMS * sizeof(bfloat16_t));
        pipe_->InitBuffer(kktQueue_, 1, KKT_ELEMS * sizeof(float));

        // Each attention head gets a fresh output tile, so KKT itself remains
        // unchanged and can be reused by the following heads.
        pipe_->InitBuffer(outputQueue_, 1, KKT_ELEMS * sizeof(float));
        pipe_->InitBuffer(gBuf_, ROW_ELEMS * sizeof(float));
        pipe_->InitBuffer(betaBuf_, ROW_ELEMS * sizeof(bfloat16_t));
        pipe_->InitBuffer(negGBuf_, ROW_ELEMS * sizeof(float));
        pipe_->InitBuffer(negativeMaskBuf_, MASK_BYTES);

        vToMte2Event_ = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
        mte2ToMEvent_ = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_M));
        mte2ToVEvent_ = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
    }

    __aicore__ inline void Process()
    {
        const int64_t workerId = static_cast<int64_t>(GetBlockIdx());

        // The generated smoke case uses equal adjacent offsets, so every
        // chunk is empty. Explicitly zero the output in that case. Each AIV
        // worker still submits one legal Matmul request before End(), which
        // keeps the MIX AIC/AIV service lifecycle balanced.
        if (!HasAnyValidChunk()) {
            if (workerId == 0) {
                const int64_t outputElems = tiling_->h * tiling_->t * BT;
                InitOutput<float>(outputGm_, outputElems, 0.0f);
            }
            RunDummyMatmul();
            mm_.End();
            return;
        }

        bool didMatmul = false;
        for (int64_t taskId = workerId;
             taskId < tiling_->totalTaskCount;
             taskId += tiling_->blockDim) {
            didMatmul = ProcessTask(taskId) || didMatmul;
        }

        // A worker may receive only empty chunks even when another worker has
        // valid work. Do not let that worker enter End() without ever issuing
        // a Matmul request.
        if (!didMatmul) {
            RunDummyMatmul();
        }
        mm_.End();
    }

private:
    __aicore__ inline bool HasAnyValidChunk()
    {
        for (int64_t chunkId = 0; chunkId < tiling_->numChunks; ++chunkId) {
            const int32_t start = chunkOffsetsGm_.GetValue(chunkId);
            const int32_t end = chunkOffsetsGm_.GetValue(chunkId + 1);
            const int32_t len = end - start;
            if (start >= 0 && end <= tiling_->t && len > 0 && len <= BT) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline void RunDummyMatmul()
    {
        // T is validated as positive on Host. One real row plus zero-padded
        // tail forms a legal fixed-shape [64,K] Matmul; its result is dropped.
        LocalTensor<bfloat16_t> kLocal = StageK(0, 0, 1);
        LocalTensor<float> dummy = kktQueue_.AllocTensor<float>();
        ComputeKkt(kLocal, dummy);
        kktQueue_.EnQue(dummy);
        LocalTensor<float> finished = kktQueue_.DeQue<float>();
        kktQueue_.FreeTensor(finished);
    }

    __aicore__ inline bool ProcessTask(int64_t taskId)
    {
        // One task owns one (chunk, kvHead). All attention heads belonging to
        // this KV head reuse the same K*K^T result.
        const int64_t kvHeadId = taskId % tiling_->hg;
        const int64_t chunkId = taskId / tiling_->hg;

        const int32_t start = chunkOffsetsGm_.GetValue(chunkId);
        const int32_t end = chunkOffsetsGm_.GetValue(chunkId + 1);
        const int32_t chunkLen = end - start;
        if (start < 0 || end > tiling_->t || chunkLen <= 0 || chunkLen > BT) {
            return false;
        }

        LocalTensor<bfloat16_t> kLocal = StageK(start, kvHeadId, chunkLen);
        LocalTensor<float> kktLocal = kktQueue_.AllocTensor<float>();
        ComputeKkt(kLocal, kktLocal);

        // Matmul writes from CO2 to VECIN. EnQue/DeQue must happen
        // immediately after GetTensorC so that Vector sees completed data.
        kktQueue_.EnQue(kktLocal);
        LocalTensor<float> readyKkt = kktQueue_.DeQue<float>();

        const int64_t firstHead = kvHeadId * tiling_->numRepeat;
        for (int32_t repeat = 0; repeat < tiling_->numRepeat; ++repeat) {
            const int64_t headId = firstHead + repeat;
            LoadGateAndBeta(headId, start, chunkLen);
            ProcessOneHead(readyKkt, headId, start, chunkLen);
        }

        kktQueue_.FreeTensor(readyKkt);
        return true;
    }

    __aicore__ inline LocalTensor<bfloat16_t> StageK(
        int32_t start,
        int64_t kvHeadId,
        int32_t chunkLen)
    {
        LocalTensor<bfloat16_t> kLocal = kBuf_.Get<bfloat16_t>();

        // Zero tail rows, so every chunk uses one fixed [64,K] Matmul shape.
        Duplicate(kLocal, ToBfloat16(0.0f), K_ELEMS);
        SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
        WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);

        // k is [B,T,Hg,K]. Rows of one KV head are separated by Hg*K.
        const int64_t kOffset =
            (static_cast<int64_t>(start) * tiling_->hg + kvHeadId) * K_DIM;
        const uint32_t rowBytes = K_DIM * sizeof(bfloat16_t);
        const uint32_t srcGapBytes =
            static_cast<uint32_t>((tiling_->hg - 1) * K_DIM * sizeof(bfloat16_t));

        DataCopyExtParams copyParams{
            static_cast<uint16_t>(chunkLen),
            rowBytes,
            srcGapBytes,
            0,
            0};
        DataCopyPadExtParams<bfloat16_t> padParams{
            false, 0, 0, ToBfloat16(0.0f)};
        DataCopyPad(kLocal, kGm_[kOffset], copyParams, padParams);

        SetFlag<HardEvent::MTE2_M>(mte2ToMEvent_);
        WaitFlag<HardEvent::MTE2_M>(mte2ToMEvent_);
        return kLocal;
    }

    __aicore__ inline void ComputeKkt(
        const LocalTensor<bfloat16_t>& kLocal,
        const LocalTensor<float>& kktLocal)
    {
        mm_.SetTensorA(kLocal, false);
        mm_.SetTensorB(kLocal, true);
        mm_.DisableBias();

        while (mm_.template Iterate<true>()) {
            mm_.template GetTensorC<true>(kktLocal, false, true);
        }
    }

    __aicore__ inline void LoadGateAndBeta(
        int64_t headId,
        int32_t start,
        int32_t chunkLen)
    {
        LocalTensor<float> gLocal = gBuf_.Get<float>();
        LocalTensor<bfloat16_t> betaLocal = betaBuf_.Get<bfloat16_t>();

        // These buffers were consumed by Vector for the previous head.
        SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
        WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);

        const int64_t headOffset = headId * tiling_->t + start;

        const uint8_t gRightPad =
            static_cast<uint8_t>((8 - (chunkLen & 7)) & 7);
        DataCopyExtParams gCopyParams{
            1,
            static_cast<uint32_t>(chunkLen * sizeof(float)),
            0,
            0,
            0};
        DataCopyPadExtParams<float> gPadParams{true, 0, gRightPad, 0.0f};
        DataCopyPad(gLocal, gGm_[headOffset], gCopyParams, gPadParams);

        const uint8_t betaRightPad =
            static_cast<uint8_t>((16 - (chunkLen & 15)) & 15);
        DataCopyExtParams betaCopyParams{
            1,
            static_cast<uint32_t>(chunkLen * sizeof(bfloat16_t)),
            0,
            0,
            0};
        DataCopyPadExtParams<bfloat16_t> betaPadParams{
            true, 0, betaRightPad, ToBfloat16(0.0f)};
        DataCopyPad(betaLocal, betaGm_[headOffset], betaCopyParams, betaPadParams);

        SetFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVEvent_);
    }

    __aicore__ inline void ProcessOneHead(
        const LocalTensor<float>& kktLocal,
        int64_t headId,
        int32_t start,
        int32_t chunkLen)
    {
        LocalTensor<float> gLocal = gBuf_.Get<float>();
        LocalTensor<bfloat16_t> betaLocal = betaBuf_.Get<bfloat16_t>();
        LocalTensor<float> negGLocal = negGBuf_.Get<float>();
        LocalTensor<uint8_t> negativeMask = negativeMaskBuf_.Get<uint8_t>();
        LocalTensor<float> outputLocal = outputQueue_.AllocTensor<float>();

        // Initialize the whole output tile once. For row i, only j<i is
        // calculated; diagonal, upper triangle and tail columns stay zero.
        Duplicate(outputLocal, 0.0f, KKT_ELEMS);
        PipeBarrier<PIPE_V>();

        // -g[j] is shared by every row of this attention head.
        Muls(negGLocal, gLocal, -1.0f, static_cast<uint32_t>(chunkLen));
        PipeBarrier<PIPE_V>();

        for (int32_t row = 1; row < chunkLen; ++row) {
            const uint32_t active = static_cast<uint32_t>(row);
            const float gi = gLocal.GetValue(row);
            const float beta = ToFloat(betaLocal.GetValue(row));

            LocalTensor<float> outputRow = outputLocal[row * ROW_ELEMS];
            LocalTensor<float> kktRow = kktLocal[row * ROW_ELEMS];

            // outputRow[j] is used as temporary storage for diff/exp and then
            // overwritten by the final KKT-weighted result.
            Adds(outputRow, negGLocal, gi, active);
            PipeBarrier<PIPE_V>();

            CompareScalar(negativeMask, outputRow, 0.0f,
                CMPMODE::LT, active);
            PipeBarrier<PIPE_V>();

            Select(outputRow, negativeMask, outputRow, NEGATIVE_INF_PROXY,
                SELMODE::VSEL_TENSOR_SCALAR_MODE, active);
            PipeBarrier<PIPE_V>();

            Exp(outputRow, outputRow, active);
            PipeBarrier<PIPE_V>();

            Select(outputRow, negativeMask, outputRow, 0.0f,
                SELMODE::VSEL_TENSOR_SCALAR_MODE, active);
            PipeBarrier<PIPE_V>();

            Muls(outputRow, outputRow, beta, active);
            PipeBarrier<PIPE_V>();

            Mul(outputRow, kktRow, outputRow, active);
            PipeBarrier<PIPE_V>();
        }

        outputQueue_.EnQue(outputLocal);
        CopyOutHead(headId, start, chunkLen);
    }

    __aicore__ inline void CopyOutHead(
        int64_t headId,
        int32_t start,
        int32_t chunkLen)
    {
        LocalTensor<float> outputLocal = outputQueue_.DeQue<float>();
        const int64_t outputOffset =
            (headId * tiling_->t + start) * static_cast<int64_t>(BT);

        DataCopy(outputGm_[outputOffset], outputLocal,
            static_cast<uint32_t>(chunkLen * ROW_ELEMS));
        outputQueue_.FreeTensor(outputLocal);
    }

private:
    MatmulObj& mm_;
    TPipe* pipe_ = nullptr;
    const ChunkScaledDotKktTilingData* tiling_ = nullptr;

    GlobalTensor<bfloat16_t> kGm_;
    GlobalTensor<bfloat16_t> betaGm_;
    GlobalTensor<float> gGm_;
    GlobalTensor<int32_t> chunkOffsetsGm_;
    GlobalTensor<float> outputGm_;

    TBuf<TPosition::VECOUT> kBuf_;
    TQue<QuePosition::VECOUT, 1> kktQueue_;
    TQue<TPosition::VECOUT, 1> outputQueue_;
    TBuf<TPosition::VECIN> gBuf_;
    TBuf<TPosition::VECIN> betaBuf_;
    TBuf<TPosition::VECCALC> negGBuf_;
    TBuf<TPosition::VECCALC> negativeMaskBuf_;

    event_t vToMte2Event_;
    event_t mte2ToMEvent_;
    event_t mte2ToVEvent_;
};

} // namespace NsChunkScaledDotKkt

#endif // CHUNK_SCALED_DOT_KKT_H