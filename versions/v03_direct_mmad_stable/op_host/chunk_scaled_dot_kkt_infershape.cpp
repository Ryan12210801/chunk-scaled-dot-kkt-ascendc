/*!
 * \file chunk_scaled_dot_kkt_infershape.cpp
 * \brief ChunkScaledDotKkt shape inference.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

namespace ops {
namespace {
constexpr int64_t CHUNK_SIZE = 64;
}

static ge::graphStatus InferShapeChunkScaledDotKkt(gert::InferShapeContext* context)
{
    const gert::Shape* kShape = context->GetInputShape(0);       // [B,T,Hg,K]
    const gert::Shape* betaShape = context->GetInputShape(1);    // [B,H,T]
    gert::Shape* outputShape = context->GetOutputShape(0);

    if (kShape == nullptr || betaShape == nullptr || outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (kShape->GetDimNum() != 4 || betaShape->GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }

    outputShape->SetDimNum(4);
    outputShape->SetDim(0, kShape->GetDim(0));       // B
    outputShape->SetDim(1, betaShape->GetDim(1));    // H
    outputShape->SetDim(2, kShape->GetDim(1));       // T
    outputShape->SetDim(3, CHUNK_SIZE);              // BT
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ChunkScaledDotKkt).InferShape(InferShapeChunkScaledDotKkt);

} // namespace ops
