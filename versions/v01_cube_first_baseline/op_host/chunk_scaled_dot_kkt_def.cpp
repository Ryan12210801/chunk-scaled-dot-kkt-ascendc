/*!
 * \file chunk_scaled_dot_kkt_def.cpp
 * \brief ChunkScaledDotKkt operator definition.
 */

#include "register/op_def_registry.h"

namespace ops {

class ChunkScaledDotKkt : public OpDef {
public:
    explicit ChunkScaledDotKkt(const char* name) : OpDef(name)
    {
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("beta")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("g_cumsum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("chunk_offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("chunk_size")
            .AttrType(OPTIONAL)
            .Int(64);

        this->Output("A")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ChunkScaledDotKkt);

} // namespace ops
