# Ascend Trace Report: trace.json

## Global summary

| metric | value |
| --- | --- |
| global span | 39.640001 µs |
| total events | 1512338 |
| complete X events | 1510838 |
| logical AI cores | 20 |
| AIC processes | 20 |
| AIV processes | 40 |
| useful AICs containing MMAD | 20 |
| control-only AICs | 0 |
| active AIVs | 40 |
| inactive AIVs | 0 |
| MMAD instructions | 314 |
| transfer bytes across all lanes | 26538496 |

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
| 0 | 1 | 2 | 0 | 39.639 | 1350272 |
| 1 | 1 | 2 | 0 | 39.638001 | 1350272 |
| 2 | 1 | 2 | 0 | 39.638001 | 1350272 |
| 3 | 1 | 2 | 0 | 39.636999 | 1350272 |
| 4 | 1 | 2 | 0 | 39.636 | 1350272 |
| 5 | 1 | 2 | 0 | 39.634001 | 1350272 |
| 6 | 1 | 2 | 0 | 39.632 | 1350272 |
| 7 | 1 | 2 | 0 | 39.631001 | 1350272 |
| 8 | 1 | 2 | 0 | 39.631001 | 1350272 |
| 9 | 1 | 2 | 0 | 39.629999 | 1350272 |
| 10 | 1 | 2 | 0 | 39.629 | 1350272 |
| 11 | 1 | 2 | 0 | 39.627001 | 1350272 |
| 12 | 1 | 2 | 0 | 39.627001 | 1313408 |
| 13 | 1 | 2 | 0 | 39.626002 | 1313408 |
| 14 | 1 | 2 | 0 | 39.624001 | 1284736 |
| 15 | 1 | 2 | 0 | 39.621001 | 1284736 |
| 16 | 1 | 2 | 0 | 39.621001 | 1284736 |
| 17 | 1 | 2 | 0 | 39.620001 | 1284736 |
| 18 | 1 | 2 | 0 | 39.619002 | 1284736 |
| 19 | 1 | 2 | 0 | 39.618001 | 1284736 |

## Active-AIV stage statistics

| stage | minimum µs | mean µs | maximum µs |
| --- | --- | --- | --- |
| device/cross-core wait | 16.298000 | 16.308325 | 16.322001 |
| post-wait to first vector math | 1.348999 | 1.362650 | 1.604002 |
| vector math window | 20.725997 | 20.973300 | 20.994998 |

## Top 30 instructions by summed lifetime

| engine | pipe | instruction | count | sum µs | avg ns | bytes |
| --- | --- | --- | --- | --- | --- | --- |
| AIV | VECTOR | MOVEMASK | 164796 | 1712.259974 | 10.39 | 0 |
| AIV | SCALAR | ST_XD_XN_IMM | 27516 | 720.497992 | 26.185 | 0 |
| AIV | FLOWCTRL | WAIT_FLAG_DEVI | 40 | 652.332998 | 16308.325 | 0 |
| AIC | SCALAR | ST_XD_XN_IMM | 18262 | 399.501999 | 21.876 | 0 |
| AIC | FIXP | FIX_L0C_TO_DST | 314 | 289.388 | 921.618 | 5120000 |
| AIV | VECTOR | VMULS | 20310 | 264.030005 | 13.0 | 0 |
| AIC | FIXP | MOV_SPR_XN | 314 | 257.984 | 821.605 | 0 |
| AIC | MTE2 | MOV_OUT_TO_L1_MULTI_ND2NZ | 628 | 249.997001 | 398.084 | 10256384 |
| AIV | VECTOR | VADDS | 20480 | 245.760002 | 12.0 | 0 |
| AIV | SCALAR | LD_XD_XN_IMM | 54512 | 245.425005 | 4.502 | 0 |
| AIC | SCALAR | STI_XN_IMM | 8376 | 238.006999 | 28.415 | 0 |
| AIC | CUBE | MMAD | 314 | 201.231002 | 640.863 | 0 |
| AIC | SCALAR | STP_XI_XJ_XN | 4902 | 187.92 | 38.335 | 0 |
| AIV | SCALAR | ADD | 166725 | 166.725008 | 1.0 | 0 |
| AIC | SCALAR | LD_XD_XN_IMM | 49908 | 161.545002 | 3.237 | 0 |
| AIV | SCALAR | LD_XD_XN | 20742 | 159.068005 | 7.669 | 0 |
| AIV | MTE3 | MOV_UB_TO_OUT | 240 | 142.179002 | 592.413 | 3932160 |
| AIV | MTE2 | MOV_SRC_TO_DST_ALIGN | 320 | 127.815 | 399.422 | 122880 |
| AIV | SCALAR | SHL | 105325 | 105.325005 | 1.0 | 0 |
| AIV | SCALAR | ZEROEXT | 94748 | 94.748005 | 1.0 | 0 |
| AIV | SCALAR | INSERT_XD | 83376 | 83.376004 | 1.0 | 0 |
| AIV | SCALAR | MOV_XD_SPR | 82697 | 82.697004 | 1.0 | 0 |
| AIV | SCALAR | STI_XN_IMM | 1521 | 73.36 | 48.231 | 0 |
| AIC | MTE3 | MOV_L1_TO_OUT | 340 | 70.416 | 207.106 | 657920 |
| AIV | SCALAR | STP_XI_XJ_XN | 440 | 65.146 | 148.059 | 0 |
| AIV | SCALAR | ADD_IMM | 52812 | 52.812003 | 1.0 | 0 |
| AIV | SCALAR | SIGNEXT | 44466 | 44.466002 | 1.0 | 0 |
| AIC | SCALAR | LD_XD_XN | 2198 | 43.258001 | 19.681 | 0 |
| AIV | VECTOR | BAR | 42308 | 42.308002 | 1.0 | 0 |
| AIV | SCALAR | CMP | 41653 | 41.653002 | 1.0 | 0 |

Summed instruction lifetimes can exceed wall-clock span because different cores and pipelines execute concurrently.

## Top 30 repeated transfer sources

| engine | instruction | src→dst | bytes/event | XN value | events | pids | total bytes | avg µs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AIC | LOAD_2D | L1→L0B | 16384 | 0x4000 | 312 | 20 | 5111808 | 0.132 |
| AIC | FIX_L0C_TO_DST | L0C32→OUT | 16384 | 0 | 312 | 20 | 5111808 | 0.926538 |
| AIV | MOV_UB_TO_OUT | UB→OUT | 16384 | 0x4000 | 80 | 40 | 1310720 | 0.235513 |
| AIV | MOV_UB_TO_OUT | UB→OUT | 16384 | 0x8000 | 80 | 40 | 1310720 | 0.349338 |
| AIC | MOV_L1_TO_OUT | L1→OUT | 2048 | 0 | 320 | 20 | 655360 | 0.207603 |
| AIV | MOV_UB_TO_OUT | UB→OUT | 16384 | 0xc000 | 40 | 40 | 655360 | 1.0901 |
| AIV | MOV_UB_TO_OUT | UB→OUT | 16384 | 0x10000 | 40 | 40 | 655360 | 1.294675 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x11311800 | 2 | 1 | 32768 | 0.5465 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x11361800 | 2 | 1 | 32768 | 0.5645 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x113b1800 | 2 | 1 | 32768 | 0.535 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x11401800 | 2 | 1 | 32768 | 0.542 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x11451800 | 2 | 1 | 32768 | 0.5455 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x114a1800 | 2 | 1 | 32768 | 0.533 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x114f1800 | 2 | 1 | 32768 | 0.545 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x11541800 | 2 | 1 | 32768 | 0.544 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x11591800 | 2 | 1 | 32768 | 0.549 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x115e1800 | 2 | 1 | 32768 | 0.5405 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x11631800 | 2 | 1 | 32768 | 0.6315 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x11681800 | 2 | 1 | 32768 | 0.5725 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x116d1800 | 2 | 1 | 32768 | 0.6295 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x11721800 | 2 | 1 | 32768 | 0.617 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x11771800 | 2 | 1 | 32768 | 0.566 |
| AIC | MOV_OUT_TO_L1_MULTI_ND2NZ | OUT→L1 | 16384 | 0x117c1800 | 2 | 1 | 32768 | 0.6035 |
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
| core0.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 21.341999 | 16.322001 | UIMM:0xd, |
| core1.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 21.341999 | 16.322001 | UIMM:0xd, |
| core1.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 21.343 | 16.320999 | UIMM:0xd, |
| core0.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 21.344 | 16.32 | UIMM:0xd, |
| core2.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 21.346001 | 16.318001 | UIMM:0xd, |
| core2.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 21.347 | 16.316999 | UIMM:0xd, |
| core3.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 21.348 | 16.316 | UIMM:0xd, |
| core4.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 21.348 | 16.316 | UIMM:0xd, |
| core3.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 21.349001 | 16.315001 | UIMM:0xd, |
| core4.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 21.35 | 16.313999 | UIMM:0xd, |
| core5.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 21.35 | 16.313999 | UIMM:0xd, |
| core5.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 21.351 | 16.313 | UIMM:0xd, |
| core6.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 21.351 | 16.313 | UIMM:0xd, |
| core6.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 21.351999 | 16.312 | UIMM:0xd, |
| core7.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 21.353001 | 16.311001 | UIMM:0xd, |
| core7.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 21.354 | 16.309999 | UIMM:0xd, |
| core8.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 21.354 | 16.309999 | UIMM:0xd, |
| core8.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 21.355 | 16.309 | UIMM:0xd, |
| core10.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 21.357 | 16.306999 | UIMM:0xd, |
| core11.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 21.357 | 16.306999 | UIMM:0xd, |
| core11.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 21.357 | 16.306999 | UIMM:0xd, |
| core9.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 21.357 | 16.306999 | UIMM:0xd, |
| core12.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 21.358 | 16.306 | UIMM:0xd, |
| core13.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 21.358 | 16.306 | UIMM:0xd, |
| core10.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 21.358999 | 16.305 | UIMM:0xd, |
| core13.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 21.358999 | 16.305 | UIMM:0xd, |
| core9.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 21.358999 | 16.305 | UIMM:0xd, |
| core12.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 21.358999 | 16.304001 | UIMM:0xd, |
| core17.veccore0 | FLOWCTRL | WAIT_FLAG_DEVI | 21.361 | 16.302999 | UIMM:0xd, |
| core14.veccore1 | FLOWCTRL | WAIT_FLAG_DEVI | 21.362 | 16.302 | UIMM:0xd, |

## Top 30 repeated PCs

| pid | pipe | instruction | PC | count | sum µs |
| --- | --- | --- | --- | --- | --- |
| core0.veccore0 | SCALAR | ST_XD_XN_IMM | 0x140e8754 | 256 | 5.376 |
| core0.veccore1 | SCALAR | ST_XD_XN_IMM | 0x140e8754 | 256 | 5.376 |
| core1.veccore0 | SCALAR | ST_XD_XN_IMM | 0x140e8754 | 256 | 5.376 |
| core1.veccore0 | SCALAR | ST_XD_XN_IMM | 0x140e87cc | 256 | 5.376 |
| core1.veccore1 | SCALAR | ST_XD_XN_IMM | 0x140e8754 | 256 | 5.376 |
| core10.veccore0 | SCALAR | ST_XD_XN_IMM | 0x140e8754 | 256 | 5.376 |
| core10.veccore1 | SCALAR | ST_XD_XN_IMM | 0x140e8754 | 256 | 5.376 |
| core11.veccore0 | SCALAR | ST_XD_XN_IMM | 0x140e8754 | 256 | 5.376 |
| core11.veccore1 | SCALAR | ST_XD_XN_IMM | 0x140e8754 | 256 | 5.376 |
| core12.veccore0 | SCALAR | ST_XD_XN_IMM | 0x140e8754 | 256 | 5.376 |
| core12.veccore1 | SCALAR | ST_XD_XN_IMM | 0x140e8754 | 256 | 5.376 |
| core13.veccore0 | SCALAR | ST_XD_XN_IMM | 0x140e8754 | 256 | 5.376 |
| core13.veccore0 | SCALAR | ST_XD_XN_IMM | 0x140e87cc | 256 | 5.376 |
| core13.veccore1 | SCALAR | ST_XD_XN_IMM | 0x140e8754 | 256 | 5.376 |
| core14.veccore0 | SCALAR | ST_XD_XN_IMM | 0x140e8754 | 256 | 5.376 |
| core14.veccore1 | SCALAR | ST_XD_XN_IMM | 0x140e8754 | 256 | 5.376 |
| core15.veccore0 | SCALAR | ST_XD_XN_IMM | 0x140e8754 | 256 | 5.376 |
| core15.veccore1 | SCALAR | ST_XD_XN_IMM | 0x140e8754 | 256 | 5.376 |
| core16.veccore0 | SCALAR | ST_XD_XN_IMM | 0x140e8754 | 256 | 5.376 |
| core16.veccore1 | SCALAR | ST_XD_XN_IMM | 0x140e8754 | 256 | 5.376 |
| core17.veccore0 | SCALAR | ST_XD_XN_IMM | 0x140e8754 | 256 | 5.376 |
| core17.veccore0 | SCALAR | ST_XD_XN_IMM | 0x140e87cc | 256 | 5.376 |
| core17.veccore1 | SCALAR | ST_XD_XN_IMM | 0x140e8754 | 256 | 5.376 |
| core18.veccore0 | SCALAR | ST_XD_XN_IMM | 0x140e8754 | 256 | 5.376 |
| core18.veccore1 | SCALAR | ST_XD_XN_IMM | 0x140e8754 | 256 | 5.376 |
| core19.veccore0 | SCALAR | ST_XD_XN_IMM | 0x140e8754 | 256 | 5.376 |
| core19.veccore1 | SCALAR | ST_XD_XN_IMM | 0x140e8754 | 256 | 5.376 |
| core19.veccore1 | SCALAR | ST_XD_XN_IMM | 0x140e87cc | 256 | 5.376 |
| core2.veccore0 | SCALAR | ST_XD_XN_IMM | 0x140e8754 | 256 | 5.376 |
| core2.veccore1 | SCALAR | ST_XD_XN_IMM | 0x140e8754 | 256 | 5.376 |