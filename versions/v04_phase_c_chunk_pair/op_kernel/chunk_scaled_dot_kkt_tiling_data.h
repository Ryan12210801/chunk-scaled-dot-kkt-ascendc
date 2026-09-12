/*!
 * \file chunk_scaled_dot_kkt_tiling_data.h
 * \brief Minimal tiling data for the split Cube/Vector implementation.
 */

#ifndef CHUNK_SCALED_DOT_KKT_TILING_DATA_H
#define CHUNK_SCALED_DOT_KKT_TILING_DATA_H

#include <cstdint>

struct ChunkScaledDotKktTilingData {
    // Values consumed directly by the device kernel.
    int64_t totalTaskCount = 0;  // numChunks * Hg
    int64_t t = 0;
    int64_t hg = 0;

    // blockDim counts AIC groups. MIX_AIC_1_2 launches 2 * blockDim AIVs.
    int32_t blockDim = 0;
    int32_t numRepeat = 0;       // H / Hg
};

static_assert(sizeof(ChunkScaledDotKktTilingData) == 32,
    "Unexpected ChunkScaledDotKktTilingData layout");

#endif // CHUNK_SCALED_DOT_KKT_TILING_DATA_H