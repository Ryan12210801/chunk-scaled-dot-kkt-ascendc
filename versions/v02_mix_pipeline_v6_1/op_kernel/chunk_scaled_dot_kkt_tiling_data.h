/*!
 * \file chunk_scaled_dot_kkt_tiling_data.h
 * \brief Compact tiling data for the split Cube/Vector implementation.
 */

#ifndef CHUNK_SCALED_DOT_KKT_TILING_DATA_H
#define CHUNK_SCALED_DOT_KKT_TILING_DATA_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct ChunkScaledDotKktTilingData {
    // Values that cannot be cheaply reconstructed in the kernel.
    int64_t totalTaskCount = 0;   // numChunks * Hg
    int64_t t = 0;
    int64_t hg = 0;
    uint64_t userWorkspaceOffset = 0;

    // blockDim counts AIC groups. MIX_AIC_1_2 launches 2 * blockDim AIVs.
    int32_t blockDim = 0;
    int32_t numRepeat = 0;        // H / Hg
    int32_t rowTileCount = 1;
    int32_t headGroupCount = 1;
    uint32_t broadcastTmpBytes = 0;  // exact minimum from host tiling

    AscendC::tiling::TCubeTiling matmulTiling;
};

// The previous 264-byte layout was rejected by the runtime tiling buffer.
// Keep the serialized structure at or below 256 bytes.
static_assert(sizeof(ChunkScaledDotKktTilingData) <= 256,
    "ChunkScaledDotKktTilingData exceeds the 256-byte tiling-data limit");

#endif // CHUNK_SCALED_DOT_KKT_TILING_DATA_H