/*!
 * \file chunk_scaled_dot_kkt_tiling.cpp
 * \brief Host tiling for the V6.1 local AIC-to-two-AIV slot pipeline.
 */

#include <algorithm>
#include <cstdint>
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/chunk_scaled_dot_kkt_tiling_data.h"
#include "../op_kernel/chunk_scaled_dot_kkt_tiling_key.h"

namespace optiling {
namespace {

constexpr int64_t B_SUPPORTED = 1;
constexpr int64_t K_128 = 128;
constexpr int64_t K_256 = 256;
constexpr int32_t SLOT_COUNT = 2;
constexpr size_t KKT_SLOT_BYTES =
    64U * 64U * sizeof(float);

ge::graphStatus ValidateAndReadShape(
    gert::TilingContext* context,
    int64_t& t,
    int64_t& h,
    int64_t& hg,
    int64_t& kDim,
    int64_t& numChunks)
{
    const gert::StorageShape* kStorage =
        context->GetInputShape(0);
    const gert::StorageShape* betaStorage =
        context->GetInputShape(1);
    const gert::StorageShape* gStorage =
        context->GetInputShape(2);
    const gert::StorageShape* offsetStorage =
        context->GetInputShape(3);

    OP_CHECK_NULL_WITH_CONTEXT(context, kStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, gStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, offsetStorage);

    const gert::Shape& kShape =
        kStorage->GetStorageShape();
    const gert::Shape& betaShape =
        betaStorage->GetStorageShape();
    const gert::Shape& gShape =
        gStorage->GetStorageShape();
    const gert::Shape& offsetShape =
        offsetStorage->GetStorageShape();

    OP_CHECK_IF(kShape.GetDimNum() != 4,
        OP_LOGE(context, "k must be rank-4 [B,T,Hg,K]"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        betaShape.GetDimNum() != 3 ||
        gShape.GetDimNum() != 3,
        OP_LOGE(
            context,
            "beta and g_cumsum must be rank-3 [B,H,T]"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(offsetShape.GetDimNum() != 1,
        OP_LOGE(
            context,
            "chunk_offsets must be rank-1"),
        return ge::GRAPH_FAILED);

    const int64_t b = kShape.GetDim(0);
    t = kShape.GetDim(1);
    hg = kShape.GetDim(2);
    kDim = kShape.GetDim(3);
    h = betaShape.GetDim(1);

    OP_CHECK_IF(b != B_SUPPORTED,
        OP_LOGE(
            context,
            "Only B=1 is supported, but got %ld",
            b),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(t <= 0 || h <= 0 || hg <= 0,
        OP_LOGE(
            context,
            "T, H and Hg must be positive"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(kDim != K_128 && kDim != K_256,
        OP_LOGE(
            context,
            "K must be 128 or 256, but got %ld",
            kDim),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(h % hg != 0,
        OP_LOGE(
            context,
            "Hg must divide H, H=%ld Hg=%ld",
            h,
            hg),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        betaShape.GetDim(0) != b ||
        betaShape.GetDim(2) != t,
        OP_LOGE(
            context,
            "beta shape must be [B,H,T]"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        gShape.GetDim(0) != b ||
        gShape.GetDim(1) != h ||
        gShape.GetDim(2) != t,
        OP_LOGE(
            context,
            "g_cumsum shape must match beta"),
        return ge::GRAPH_FAILED);

    const int64_t offsetCount = offsetShape.GetDim(0);
    OP_CHECK_IF(offsetCount < 2,
        OP_LOGE(
            context,
            "chunk_offsets must contain at least two values"),
        return ge::GRAPH_FAILED);

    numChunks = offsetCount - 1;
    return ge::GRAPH_SUCCESS;
}


} // namespace

static ge::graphStatus ChunkScaledDotKktTilingFunc(
    gert::TilingContext* context)
{
    fe::PlatFormInfos* platformInfo =
        context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    const auto platform =
        platform_ascendc::PlatformAscendC(platformInfo);
    const int64_t aicNum =
        static_cast<int64_t>(
            platform.GetCoreNumAic());

    OP_CHECK_IF(aicNum <= 0,
        OP_LOGE(
            context,
            "AIC core number must be positive"),
        return ge::GRAPH_FAILED);

    int64_t t = 0;
    int64_t h = 0;
    int64_t hg = 0;
    int64_t kDim = 0;
    int64_t numChunks = 0;

    OP_CHECK_IF(
        ValidateAndReadShape(
            context,
            t,
            h,
            hg,
            kDim,
            numChunks) != ge::GRAPH_SUCCESS,
        OP_LOGE(
            context,
            "Input validation failed"),
        return ge::GRAPH_FAILED);

    ChunkScaledDotKktTilingData* tiling =
        context->GetTilingData<
            ChunkScaledDotKktTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const int64_t totalTaskCount =
        numChunks * hg;

    // Every selected AIC group owns at least one Cube task and is paired with
    // exactly two local AIV consumers. No global all-core barrier is required.
    const int32_t blockDim =
        static_cast<int32_t>(
            std::min<int64_t>(
                aicNum,
                totalTaskCount));

    tiling->totalTaskCount = totalTaskCount;
    tiling->t = t;
    tiling->hg = hg;
    tiling->blockDim = blockDim;
    tiling->numRepeat =
        static_cast<int32_t>(h / hg);

    // Mode-2 AIC/AIV flags require the three slices of every selected MIX
    // group to be resident together.
    OP_CHECK_IF(
        context->SetScheduleMode(1U) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(
            context,
            "Set batch schedule mode failed"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        context->SetBlockDim(blockDim) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(
            context,
            "Set blockDim failed, blockDim=%d",
            blockDim),
        return ge::GRAPH_FAILED);

    context->SetTilingKey(
        kDim == K_128
            ? GET_TPL_TILING_KEY(
                CHUNKSCALEDDOTKKT_TPL_K_128)
            : GET_TPL_TILING_KEY(
                CHUNKSCALEDDOTKKT_TPL_K_256));

    const size_t sysWorkspace =
        static_cast<size_t>(
            platform.GetLibApiWorkSpaceSize());
    const size_t userWorkspace =
        static_cast<size_t>(blockDim) *
        SLOT_COUNT *
        KKT_SLOT_BYTES;

    size_t* workspaceSizes =
        context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        workspaceSizes);
    workspaceSizes[0] =
        sysWorkspace + userWorkspace;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForChunkScaledDotKkt(
    [[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ChunkScaledDotKktCompileInfo {};

IMPL_OP_OPTILING(ChunkScaledDotKkt)
    .Tiling(ChunkScaledDotKktTilingFunc)
    .TilingParse<ChunkScaledDotKktCompileInfo>(
        TilingParseForChunkScaledDotKkt);

} // namespace optiling