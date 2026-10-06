# Stack bench summary (2026-10-01)

HiGHS warm API: changeColsCost/Bounds + changeRowsBounds; getBasis/setBasis; threads=1; workers<=2; --no-gpu.

| level | kind | S | highs_cold wall/sum | highs_warm wall/sum | qenivo_cold wall/sum | qenivo_warm wall/sum | gpu | zp% q-warm | max |obj| rel |
|---|---|---:|---|---|---|---|---|---:|---:|
| L2 | independent | 4 | 3.20/0.30 | 3.14/0.26 | 3.33/0.63 | 2.79/0.39 | skipped_no_gpu_flag | 0.0 | 1.4881739204897276e-15 |
| L2 | one_crude | 4 | 3.01/0.30 | 2.62/0.18 | 10.70/9.38 | 2.90/0.65 | skipped_no_gpu_flag | 0.0 | 4.39463654175156e-16 |
| L3 | cargo_menu | 64 | 38.38/71.04 | 5.57/4.02 | 62.65/119.81 | 4.47/3.96 | skipped_no_gpu_flag | 57.8 | 1.6324956974977493e-11 |
| L3 | cargo_menu | 256 | 148.39/289.12 | 9.19/9.08 | 230.60/454.36 | 5.97/6.72 | skipped_no_gpu_flag | 64.5 | 2.172373057204026e-11 |
| L3 | factor_mixed | 64 | 37.06/68.60 | 7.55/8.95 | 53.00/101.19 | 13.83/18.93 | skipped_no_gpu_flag | 0.0 | 2.4910224430350363e-11 |
| L3 | factor_mixed | 256 | 136.52/266.05 | 20.50/33.04 | 225.54/445.35 | 39.90/64.80 | skipped_no_gpu_flag | 0.0 | 2.6783584513730644e-11 |
| L3 | factor_price | 64 | 37.39/68.35 | 6.37/5.97 | 56.38/108.10 | 6.03/6.06 | skipped_no_gpu_flag | 1.6 | 3.5462715257979575e-12 |
| L3 | factor_price | 256 | 139.21/270.09 | 11.87/16.55 | 202.18/398.13 | 8.12/9.38 | skipped_no_gpu_flag | 5.5 | 4.3100514385446364e-12 |
| L3 | independent | 64 | 20.25/35.55 | 20.11/33.23 | 44.82/84.68 | 33.25/60.86 | skipped_no_gpu_flag | 0.0 | 2.947749202141304e-15 |
| L3 | independent | 256 | 81.75/156.19 | 73.93/140.23 | 170.78/336.76 | 127.14/247.19 | skipped_no_gpu_flag | 0.0 | 3.806244839696342e-15 |
| L3 | one_crude | 64 | 41.81/77.27 | 5.21/3.99 | 59.89/114.69 | 5.46/4.58 | skipped_no_gpu_flag | 57.8 | 1.6324956974977493e-11 |
| L3 | one_crude | 256 | 158.18/308.48 | 7.54/7.24 | 215.09/426.26 | 6.22/6.86 | skipped_no_gpu_flag | 64.5 | 2.172373057204026e-11 |
| L3 | one_unit | 64 | 41.73/76.92 | 4.95/3.83 | 57.33/109.62 | 5.71/5.22 | skipped_no_gpu_flag | 26.6 | 9.208597998470615e-16 |
| L3 | one_unit | 256 | 158.30/308.96 | 7.90/6.41 | 214.07/422.95 | 6.95/6.95 | skipped_no_gpu_flag | 43.4 | 9.205396342341831e-16 |
