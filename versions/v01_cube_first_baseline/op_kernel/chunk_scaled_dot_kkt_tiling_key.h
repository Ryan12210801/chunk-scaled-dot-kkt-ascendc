/*!
 * \file chunk_scaled_dot_kkt_tiling_key.h
 * \brief Compile-time K selection and MIX kernel type.
 */

#ifndef CHUNK_SCALED_DOT_KKT_TILING_KEY_H
#define CHUNK_SCALED_DOT_KKT_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define CHUNKSCALEDDOTKKT_TPL_K_128 0
#define CHUNKSCALEDDOTKKT_TPL_K_256 1

ASCENDC_TPL_ARGS_DECL(
    ChunkScaledDotKkt,
    ASCENDC_TPL_UINT_DECL(kMode, 1, ASCENDC_TPL_UI_LIST,
        CHUNKSCALEDDOTKKT_TPL_K_128,
        CHUNKSCALEDDOTKKT_TPL_K_256));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_1),
        ASCENDC_TPL_UINT_SEL(kMode, ASCENDC_TPL_UI_LIST,
            CHUNKSCALEDDOTKKT_TPL_K_128,
            CHUNKSCALEDDOTKKT_TPL_K_256)));

#endif // CHUNK_SCALED_DOT_KKT_TILING_KEY_H
