/*
 * \file chunk_scaled_dot_kkt.cpp
 * \brief ChunkScaledDotKkt kernel entry.
 */

#include "chunk_scaled_dot_kkt.h"
#include "chunk_scaled_dot_kkt_tiling_key.h"

using namespace AscendC;

__aicore__ inline bool HasValidChunk(
    GM_ADDR chunkOffsets,
    const ChunkScaledDotKktTilingData& tilingData)
{
    GlobalTensor<int32_t> offsetsGm;
    offsetsGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(chunkOffsets),
        tilingData.numChunks + 1);

    for (int64_t chunkId = 0; chunkId < tilingData.numChunks; ++chunkId) {
        const int32_t start = offsetsGm.GetValue(chunkId);
        const int32_t end = offsetsGm.GetValue(chunkId + 1);
        const int32_t chunkLen = end - start;
        if (start >= 0 && end <= tilingData.t &&
            chunkLen > 0 && chunkLen <= 64) {
            return true;
        }
    }
    return false;
}

template <uint32_t kMode>
__global__ __aicore__ void chunk_scaled_dot_kkt(
    GM_ADDR k,
    GM_ADDR beta,
    GM_ADDR gCumsum,
    GM_ADDR chunkOffsets,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    // REGIST_MATMUL_OBJ requires the kernel-side Cube object configuration.
    // The tiling template still selects MIX_AIC_1_1 for generated variants;
    // this declaration supplies ASCubeObjConfig required by CANN 9.0 macros.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);

    REGISTER_TILING_DEFAULT(ChunkScaledDotKktTilingData);
    GET_TILING_DATA_WITH_STRUCT(ChunkScaledDotKktTilingData, tilingData, tiling);

    // The generated smoke case may contain only empty chunks. Avoid entering
    // the Matmul MIX registration/lifecycle when no Iterate call can occur.
    if (!HasValidChunk(chunkOffsets, tilingData)) {
        return;
    }

    SetSysWorkspace(workspace);
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }

    TPipe pipe;

    if constexpr (kMode == CHUNKSCALEDDOTKKT_TPL_K_128) {
        using Kernel = NsChunkScaledDotKkt::ChunkScaledDotKkt<128>;
        typename Kernel::MatmulObj mm;
        REGIST_MATMUL_OBJ(
            &pipe, GetSysWorkSpacePtr(), mm, &tilingData.matmulTiling);
        Kernel op(mm, &pipe);
        op.Init(k, beta, gCumsum, chunkOffsets, output, &tilingData);
        op.Process();
    }

    if constexpr (kMode == CHUNKSCALEDDOTKKT_TPL_K_256) {
        using Kernel = NsChunkScaledDotKkt::ChunkScaledDotKkt<256>;
        typename Kernel::MatmulObj mm;
        REGIST_MATMUL_OBJ(
            &pipe, GetSysWorkSpacePtr(), mm, &tilingData.matmulTiling);
        Kernel op(mm, &pipe);
        op.Init(k, beta, gCumsum, chunkOffsets, output, &tilingData);
        op.Process();
    }
}