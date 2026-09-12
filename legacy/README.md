# Legacy material

原工作区的 `tests/ut` 是早期框架 UT 模板。它包含 TODO，并引用旧版本 tiling 字段，不能与当前发布基线直接配套；为避免误用和引入不必要的第三方测试依赖，本发布包没有复制这些文件。

当前维护中的验证入口是 `../examples/test_aclnn_chunk_scaled_dot_kkt.cpp` 和 `../scripts/run_kkt.sh`。

如需考古，原始文件仍位于发布包之外的 `best_chunk_scaled_dot_kkt_real/tests/ut`；它们应先适配当前 tiling ABI，再作为独立变更引入。
