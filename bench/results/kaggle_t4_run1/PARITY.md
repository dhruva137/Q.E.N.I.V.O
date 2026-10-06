# Coverage (parity) evidence

Every counted answer was proven by the engine AND re-checked by the independent verifier.

| set | instances | proven + verified | optimal & agrees with reference | infeasible with Farkas proof | HiGHS solved (same limits) |
|---|---|---|---|---|---|
| kennington | 16 | 14 | 7 | 0 | 16 |
| maros | 135 | 96 | 94 | 0 | 100 |
| netlib | 98 | 89 | 80 | 0 | 93 |
| netlib_infeas | 29 | 23 | 0 | 23 | 29 |

**kennington**: engines {'pdhg/numpy': 4, 'pdhg/cupy': 8, 'ipm/numpy': 2}; not verified: ken-13 ken-18
**maros**: engines {'pdqp/numpy': 93, 'pdqp/cupy': 3}; not verified: BOYD1 CONT-101 CONT-200 CONT-201 CONT-300 CVXQP1_L CVXQP2_L CVXQP3_L CVXQP3_M DTOC3 HS268 HUES-MOD HUESTIS KSIP LISWET1 LISWET10 LISWET11 LISWET12 LISWET2 LISWET3 LISWET4 LISWET6 LISWET7 LISWET8 LISWET9 POWELL20 Q25FV47 QBEACONF QCAPRI QGROW15 QGROW22 QGROW7 QPCBOEI1 QPILOTNO QSHARE2B S268 STADAT1 STADAT2 STADAT3
**netlib**: engines {'ipm/numpy': 24, 'simplex/numpy': 60, 'pdhg/numpy': 4, 'pdhg/cupy': 1}; not verified: 80bau3b bnl2 forplan greenbea greenbeb pilot pilot.ja pilot87 pilotnov
**netlib_infeas**: engines {'simplex/numpy': 18, 'pdhg/numpy': 3, 'ipm/numpy': 2}; not verified: bgindy cplex2 gran greenbea-infeas pang refinery

commit  · GPU {'gpu': 'Tesla T4', 'gpu_mem_bytes': 15636037632, 'compute_capability': '7.5', 'cuda_runtime': 12090, 'cupy': '14.0.1'}