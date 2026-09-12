/*!
 * \file chunk_scaled_dot_kkt.cpp
 * \brief V6.1 ping-pong slot-pipelined MIX 1:2 kernel entry.
 */

#include "chunk_scaled_dot_kkt.h"
#include "chunk_scaled_dot_kkt_tiling_key.h"

using namespace AscendC;

namespace {

template <uint32_t kMode>
__aicore__ inline void RunCubeStage(
    GM_ADDR k,
    GM_ADDR chunkOffsets,
    GM_ADDR workspace,
    const ChunkScaledDotKktTilingData& tilingData,
    TPipe* pipe)
{
    if constexpr (kMode == CHUNKSCALEDDOTKKT_TPL_K_128) {
        using CubeStage =
            NsChunkScaledDotKkt::ChunkScaledDotKktCubeStage<128>;
        typename CubeStage::MatmulObj mm;

        REGIST_MATMUL_OBJ(
            pipe,
            GetSysWorkSpacePtr(),
            mm,
            &tilingData.matmulTiling);

        CubeStage cube(mm);
        cube.Init(
            k,
            chunkOffsets,
            workspace,
            &tilingData);
        cube.Process();
    }

    if constexpr (kMode == CHUNKSCALEDDOTKKT_TPL_K_256) {
        using CubeStage =
            NsChunkScaledDotKkt::ChunkScaledDotKktCubeStage<256>;
        typename CubeStage::MatmulObj mm;

        REGIST_MATMUL_OBJ(
            pipe,
            GetSysWorkSpacePtr(),
            mm,
            &tilingData.matmulTiling);

        CubeStage cube(mm);
        cube.Init(
            k,
            chunkOffsets,
            workspace,
            &tilingData);
        cube.Process();
    }
}

__aicore__ inline void RunVectorStage(
    GM_ADDR beta,
    GM_ADDR gCumsum,
    GM_ADDR chunkOffsets,
    GM_ADDR output,
    GM_ADDR workspace,
    const ChunkScaledDotKktTilingData& tilingData,
    TPipe* pipe)
{
    NsChunkScaledDotKkt::ChunkScaledDotKktVectorStage vector(pipe);
    vector.Init(
        beta,
        gCumsum,
        chunkOffsets,
        output,
        workspace,
        &tilingData);
    vector.Process();
}

} // namespace

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
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    REGISTER_TILING_DEFAULT(ChunkScaledDotKktTilingData);
    GET_TILING_DATA_WITH_STRUCT(
        ChunkScaledDotKktTilingData,
        tilingData,
        tiling);

    TPipe pipe;

    Async<EngineType::AIC, RunCubeStage<kMode>>(
        k,
        chunkOffsets,
        workspace,
        tilingData,
        &pipe);

    Async<EngineType::AIV, RunVectorStage>(
        beta,
        gCumsum,
        chunkOffsets,
        output,
        workspace,
        tilingData,
        &pipe);
}