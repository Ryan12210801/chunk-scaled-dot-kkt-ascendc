# Ascend Trace Report: trace_2.json

## Global summary

| metric | value |
| --- | --- |
| global span | 40.520001 µs |
| total events | 948564 |
| complete X events | 947064 |
| logical AI cores | 20 |
| AIC processes | 20 |
| AIV processes | 40 |
| useful AICs containing MMAD | 20 |
| control-only AICs | 0 |
| active AIVs | 40 |
| inactive AIVs | 0 |
| MMAD instructions | 314 |
| transfer bytes across all lanes | 31851264 |

`cubecore` identifies the process that runs on the AIC side. Only instructions on the `CUBE` lane, such as `MMAD`, consume the Cube matrix engine. Scalar, MTE, FIXP, and FLOWCTRL instructions on the same process are not Cube arithmetic.

Chrome Trace numeric `ts`/`dur` fields are treated as microseconds. `displayTimeUnit: ns` only controls viewer presentation.

## Process roles

| role | count |
| --- | --- |
| active_aiv | 40 |
| useful_aic | 20 |

## Logical-core layout

| core | useful AIC | active AIV | inactive AIV | span µs | transfer bytes |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 2 | 0 | 40.520001 | 1618560 |
| 1 | 1 | 2 | 0 | 40.518997 | 1618560 |
| 2 | 1 | 2 | 0 | 40.517999 | 1617024 |
| 3 | 1 | 2 | 0 | 40.518002 | 1618560 |
| 4 | 1 | 2 | 0 | 40.514998 | 1618560 |
| 5 | 1 | 2 | 0 | 40.515002 | 1618304 |
| 6 | 1 | 2 | 0 | 40.513001 | 1617024 |
| 7 | 1 | 2 | 0 | 40.513001 | 1617024 |
| 8 | 1 | 2 | 0 | 40.511 | 1617024 |
| 9 | 1 | 2 | 0 | 40.511002 | 1617024 |
| 10 | 1 | 2 | 0 | 40.509001 | 1618560 |
| 11 | 1 | 2 | 0 | 40.509002 | 1617792 |
| 12 | 1 | 2 | 0 | 40.508002 | 1581696 |
| 13 | 1 | 2 | 0 | 40.506001 | 1580160 |
| 14 | 1 | 2 | 0 | 40.504002 | 1551488 |
| 15 | 1 | 2 | 0 | 40.502998 | 1517952 |
| 16 | 1 | 2 | 0 | 40.503001 | 1551488 |
| 17 | 1 | 2 | 0 | 40.500998 | 1551488 |
| 18 | 1 | 2 | 0 | 40.5 | 1551488 |
| 19 | 1 | 2 | 0 | 40.499003 | 1551488 |

## Active-AIV stage statistics

| stage | minimum µs | mean µs | maximum µs |
| --- | --- | --- | --- |
| device/cross-core wait | 16.462999 | 16.482175 | 16.507000 |
| post-wait to first vector math | 2.249001 | 2.263299 | 2.446999 |
| vector math window | 17.969003 | 20.088101 | 20.577999 |

## Top 30 instructions by summed lifetime

| engine | pipe | instruction | count | sum µs | avg ns | bytes |
| --- | --- | --- | --- | --- | --- | --- |
| AIV | FLOWCTRL | WAIT_FLAG_DEVI | 40 | 659.287001 | 16482.175 | 0 |
| AIV | SCALAR | ST_XD_XN_IMM | 19675 | 647.574995 | 32.914 | 0 |
| AIV | VECTOR | MOVEMASK | 40500 | 420.887985 | 10.392 | 0 |
| AIC | SCALAR | ST_XD_XN_IMM | 18262 | 402.422999 | 22.036 | 0 |
| AIV | VECTOR | VOR | 4064 | 305.872003 | 75.264 | 0 |
| AIC | FIXP | FIX_L0C_TO_DST | 314 | 291.039001 | 926.876 | 5120000 |
| AIC | FIXP | MOV_SPR_XN | 314 | 258.869 | 824.424 | 0 |
| AIC | SCALAR | STI_XN_IMM | 8376 | 249.24 | 29.756 | 0 |
| AIC | MTE2 | MOV_OUT_TO_L1_MULTI_ND2NZ | 628 | 248.464001 | 395.643 | 10256384 |
| AIV | MTE2 | MOV_SRC_TO_DST_ALIGN | 587 | 241.121 | 410.768 | 225536 |
| AIV | SCALAR | LD_XD_XN_IMM | 72818 | 230.497002 | 3.165 | 0 |
| AIC | CUBE | MMAD | 314 | 201.853998 | 642.847 | 0 |
| AIV | MTE3 | MOV_UB_TO_OUT | 479 | 193.574 | 404.121 | 7847936 |
| AIC | SCALAR | STP_XI_XJ_XN | 4902 | 184.198998 | 37.576 | 0 |
| AIV | SCALAR | STP_XI_XJ_XN | 1440 | 183.995 | 127.774 | 0 |
| AIC | SCALAR | LD_XD_XN_IMM | 50148 | 171.992002 | 3.43 | 0 |
| AIV | SCALAR | STI_XN_IMM | 2947 | 141.061999 | 47.866 | 0 |
| AIV | VECTOR | VCOPY | 8111 | 105.443002 | 13.0 | 0 |
| AIV | VECTOR | VMUL | 966 | 91.733998 | 94.963 | 0 |
| AIC | MTE3 | MOV_L1_TO_OUT | 340 | 76.625 | 225.368 | 657920 |
| AIV | VECTOR | VBRCB | 8122 | 73.097997 | 9.0 | 0 |
| AIV | SCALAR | MOV_XD_IMM | 64753 | 64.753003 | 1.0 | 0 |
| AIV | SCALAR | ADD | 57362 | 57.362003 | 1.0 | 0 |
| AIV | SCALAR | STI_XN_XM | 1424 | 55.044 | 38.654 | 0 |
| AIV | SCALAR | ZEROEXT | 51311 | 51.311002 | 1.0 | 0 |
| AIV | MTE2 | MOV_OUT_TO_UB | 199 | 47.828 | 240.342 | 2615296 |
| AIC | SCALAR | LDP_XI_XJ_XN | 4856 | 47.703 | 9.824 | 0 |
| AIC | SCALAR | LD_XD_XN | 2198 | 43.073001 | 19.596 | 0 |
| AIV | SCALAR | SHL | 42253 | 42.253002 | 1.0 | 0 |
| AIC | MTE1 | LOAD_2D | 314 | 41.34 | 131.656 | 5128192 |

Summed instruction lifetimes can exceed wall-clock span because different cores and pipelines execute concurrently.

## Top 30 repeated transfer sources

| engine | instruction | src→dst | bytes/event | XN value | events | pids | total bytes | avg µs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AIC | LOAD_2D | L1→L0B | 16384 | 0x4000 | 312 | 20 | 5111808 | 0.132 |
| AIC | FIX_L0C_TO_DST | L0C32→OUT | 16384 | 0 | 312 | 20 | 5111808 | 0.93183 |
| AIV | MOV_UB_TO_OUT | UB→OUT | 16384 | 0xc000 | 120 | 40 | 1966080 | 0.27055 |
| AIV | MOV_UB_TO_OUT | UB→OUT | 16384 | 0x10000 | 120 | 40 | 1966080 | 0.4107 |
| AIV | MOV_UB_TO_OUT | UB→OUT | 16384 | 0x14000 | 120 | 40 | 1966080 | 0.315725 |
| AIV | MOV_UB_TO_OUT | UB→OUT | 16384 | 0x18000 | 119 | 40 | 1949696 | 0.621319 |
| AIC | MOV_L1_TO_OUT | L1→OUT | 2048 | 0 | 320 | 20 | 655360 | 0.225841 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x11311800 | 2 | 1 | 32768 | 0.5335 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x11361800 | 2 | 1 | 32768 | 0.567 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x113b1800 | 2 | 1 | 32768 | 0.5405 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x11401800 | 2 | 1 | 32768 | 0.553 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x11451800 | 2 | 1 | 32768 | 0.537 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x114a1800 | 2 | 1 | 32768 | 0.536 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x114f1800 | 2 | 1 | 32768 | 0.556 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x11541800 | 2 | 1 | 32768 | 0.5385 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x11591800 | 2 | 1 | 32768 | 0.5375 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x115e1800 | 2 | 1 | 32768 | 0.554 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x11631800 | 2 | 1 | 32768 | 0.637 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x11681800 | 2 | 1 | 32768 | 0.602 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x116d1800 | 2 | 1 | 32768 | 0.6285 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x11721800 | 2 | 1 | 32768 | 0.63 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x11771800 | 2 | 1 | 32768 | 0.577 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x117c1800 | 2 | 1 | 32768 | 0.6255 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x11311900 | 2 | 1 | 32768 | 0.236 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x11361900 | 2 | 1 | 32768 | 0.236 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x113b1900 | 2 | 1 | 32768 | 0.236 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x11401900 | 2 | 1 | 32768 | 0.236 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x11451900 | 2 | 1 | 32768 | 0.236 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x114a1900 | 2 | 1 | 32768 | 0.236 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x114f1900 | 2 | 1 | 32768 | 0.236 |

For many OUT→UB instructions, `XN` is the observed OUT-side address. The report preserves raw register values because exact register roles can vary by instruction.

## Longest 30 waits

| pid | pipe | wait | start µs | duration µs | detail |
| --- | --- | --- | --- | --- | --- |
| core0.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 23.797001 | 16.507 | UIMM:0xd, |
| core0.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 23.798 | 16.506001 | UIMM:0xd, |
| core1.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 23.798 | 16.506001 | UIMM:0xd, |
| core1.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 23.801001 | 16.502001 | UIMM:0xd, |
| core2.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 23.802999 | 16.500999 | UIMM:0xd, |
| core4.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 23.802999 | 16.5 | UIMM:0xd, |
| core2.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 23.804001 | 16.499001 | UIMM:0xd, |
| core3.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 23.805 | 16.497999 | UIMM:0xd, |
| core4.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 23.806999 | 16.497 | UIMM:0xd, |
| core5.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 23.806999 | 16.497 | UIMM:0xd, |
| core6.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 23.806 | 16.497 | UIMM:0xd, |
| core3.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 23.808001 | 16.496 | UIMM:0xd, |
| core5.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 23.806999 | 16.496 | UIMM:0xd, |
| core6.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 23.811001 | 16.492001 | UIMM:0xd, |
| core7.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 23.818001 | 16.485001 | UIMM:0xd, |
| core7.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 23.819 | 16.483999 | UIMM:0xd, |
| core13.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 23.820999 | 16.482 | UIMM:0xd, |
| core13.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 23.823999 | 16.479 | UIMM:0xd, |
| core8.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 23.825001 | 16.478001 | UIMM:0xd, |
| core12.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 23.827 | 16.476999 | UIMM:0xd, |
| core8.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 23.827999 | 16.476 | UIMM:0xd, |
| core9.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 23.827 | 16.476 | UIMM:0xd, |
| core11.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 23.827999 | 16.475 | UIMM:0xd, |
| core11.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 23.829 | 16.474001 | UIMM:0xd, |
| core9.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 23.829 | 16.474001 | UIMM:0xd, |
| core12.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 23.83 | 16.473 | UIMM:0xd, |
| core10.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 23.832001 | 16.472 | UIMM:0xd, |
| core14.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 23.830999 | 16.472 | UIMM:0xd, |
| core15.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 23.832001 | 16.471001 | UIMM:0xd, |
| core17.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 23.832001 | 16.471001 | UIMM:0xd, |

## Top 30 repeated PCs

| pid | pipe | instruction | PC | count | sum µs |
| --- | --- | --- | --- | --- | --- |
| core0.veccore0 | VECTOR | VCOPY | 0x140e954c | 56 | 0.728 |
| core0.veccore0 | VECTOR | VCOPY | 0x140e9c14 | 56 | 0.728 |
| core0.veccore1 | VECTOR | VCOPY | 0x140e954c | 56 | 0.728 |
| core0.veccore1 | VECTOR | VCOPY | 0x140e9c14 | 56 | 0.728 |
| core0.veccore1 | VECTOR | VCOPY | 0x140ea154 | 56 | 0.728 |
| core1.veccore0 | VECTOR | VCOPY | 0x140e954c | 56 | 0.728 |
| core1.veccore0 | VECTOR | VCOPY | 0x140e9c14 | 56 | 0.728 |
| core1.veccore0 | VECTOR | VCOPY | 0x140ea154 | 56 | 0.728 |
| core1.veccore1 | VECTOR | VCOPY | 0x140e954c | 56 | 0.728 |
| core1.veccore1 | VECTOR | VCOPY | 0x140e9c14 | 56 | 0.728 |
| core1.veccore1 | VECTOR | VCOPY | 0x140ea154 | 56 | 0.728 |
| core10.veccore0 | VECTOR | VCOPY | 0x140e954c | 56 | 0.728 |
| core10.veccore0 | VECTOR | VCOPY | 0x140e9c14 | 56 | 0.728 |
| core10.veccore0 | VECTOR | VCOPY | 0x140ea154 | 56 | 0.728 |
| core10.veccore1 | VECTOR | VCOPY | 0x140e954c | 56 | 0.728 |
| core10.veccore1 | VECTOR | VCOPY | 0x140e9c14 | 56 | 0.728 |
| core10.veccore1 | VECTOR | VCOPY | 0x140ea154 | 56 | 0.728 |
| core11.veccore0 | VECTOR | VCOPY | 0x140e954c | 56 | 0.728 |
| core11.veccore0 | VECTOR | VCOPY | 0x140e9c14 | 56 | 0.728 |
| core11.veccore0 | VECTOR | VCOPY | 0x140ea154 | 56 | 0.728 |
| core12.veccore0 | VECTOR | VCOPY | 0x140e954c | 56 | 0.728 |
| core12.veccore0 | VECTOR | VCOPY | 0x140e9c14 | 56 | 0.728 |
| core12.veccore0 | VECTOR | VCOPY | 0x140ea154 | 56 | 0.728 |
| core12.veccore1 | VECTOR | VCOPY | 0x140e954c | 56 | 0.728 |
| core12.veccore1 | VECTOR | VCOPY | 0x140e9c14 | 56 | 0.728 |
| core3.veccore0 | VECTOR | VCOPY | 0x140e954c | 56 | 0.728 |
| core3.veccore0 | VECTOR | VCOPY | 0x140e9c14 | 56 | 0.728 |
| core3.veccore1 | VECTOR | VCOPY | 0x140e954c | 56 | 0.728 |
| core3.veccore1 | VECTOR | VCOPY | 0x140e9c14 | 56 | 0.728 |
| core4.veccore0 | VECTOR | VCOPY | 0x140e954c | 56 | 0.728 |