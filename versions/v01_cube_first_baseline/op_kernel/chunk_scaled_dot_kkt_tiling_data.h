/*!
 * \file chunk_scaled_dot_kkt_tiling_data.h
 * \brief ChunkScaledDotKkt tiling data.
 */

#ifndef CHUNK_SCALED_DOT_KKT_TILING_DATA_H
#define CHUNK_SCALED_DOT_KKT_TILING_DATA_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct ChunkScaledDotKktTilingData {
    // V1 task: one (chunk, attention-head) pair.
    int64_t totalTaskCount = 0;
    int64_t t = 0;
    int64_t h = 0;
    int64_t hg = 0;
    int64_t numChunks = 0;

    int32_t blockDim = 0;
    int32_t numRepeat = 0;  // H / Hg
    int32_t chunkSize = 64;
    int32_t kDim = 0;

    AscendC::tiling::TCubeTiling matmulTiling;
};

#endif // CHUNK_SCALED_DOT_KKT_TILING_DATA_H
