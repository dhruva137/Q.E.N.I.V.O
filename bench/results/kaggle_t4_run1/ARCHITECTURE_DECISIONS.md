# Architecture decisions (from measurements)

## Router: fastest engine among verified answers, by model size

| size band | wins per engine |
|---|---|
| 0-600 rows | {'ipm/numpy': 13, 'pdhg/numpy': 4, 'simplex/numpy': 4} |
| 600-3000 rows | {'ipm/numpy': 7, 'pdhg/numpy': 2} |
| 3000-20000 rows | {'pdhg/cupy': 1} |

Use: set RoutingPolicy thresholds to the band edges where the winner changes.

## PDHG design ablation (1e-6, GPU): solved count and median time over solved

| config | solved | median time (s) | median iterations |
|---|---|---|---|
| default | 2/3 | 19.18 | 76832 |
| no_restarts | 0/3 | nan | nan |
| no_halpern | 1/3 | 107.92 | 319040 |
| no_reflection | 2/3 | 30.69 | 122944 |
| no_scaling | 0/3 | nan | nan |
| check_32 | 2/3 | 22.86 | 73840 |
| check_256 | 2/3 | 11.67 | 58368 |
| vendor_kernels | 2/2 | 33.30 | 76832 |
