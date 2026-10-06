# Same machine, same instances, same 1e-6 target

| instance | nirnay | highs_simplex | highs_ipm | highs_pdlp | ortools_glop | ortools_pdlp | scip | cupdlpx | cuopt |
|---|---|---|---|---|---|---|---|---|---|
| 25fv47 | optimal 0.7s | Optimal 0.2s | Optimal 0.2s | Optimal 3.8s | OPTIMAL 0.2s | OPTIMAL 2.1s | optimal 0.7s | REDUCED 3.2s | n/a |
| 80bau3b | optimal 83.0s | Optimal 0.2s | Optimal 0.3s | Optimal 48.8s | OPTIMAL 0.3s | OPTIMAL 12.7s | optimal 0.5s | REDUCED 5.1s | n/a |
| d2q06c | optimal 9.6s | Optimal 1.0s | Optimal 0.6s | Optimal 185.4s | OPTIMAL 1.0s | OPTIMAL 45.5s | optimal 4.0s | REDUCED 112.8s | n/a |
| degen3 | optimal 1.7s | Optimal 0.2s | Optimal 0.2s | Optimal 2.7s | OPTIMAL 0.3s | OPTIMAL 1.7s | optimal 1.0s | REDUCED 1.2s | n/a |
| dfl001 | optimal 10.4s | Optimal 7.7s | Optimal 2.1s | Optimal 17.5s | OPTIMAL 6.0s | OPTIMAL 8.1s | optimal 25.0s | REDUCED 2.5s | n/a |
| fit2p | optimal 18.0s | Optimal 1.2s | Optimal 0.4s | Unknown 16.4s | OPTIMAL 0.7s | OPTIMAL 11.2s | optimal 1.4s | REDUCED 1.1s | n/a |
| greenbea | not_proven 483.6s | Optimal 0.3s | Optimal 0.3s | Time limit 240.1s | OPTIMAL 0.8s | NOT_SOLVED 240.0s | optimal 4.3s | REDUCED 74.5s | n/a |
| maros-r7 | optimal 2.8s | Optimal 0.9s | Optimal 0.6s | Optimal 4.0s | OPTIMAL 0.9s | OPTIMAL 2.4s | optimal 1.2s | REDUCED 1.6s | n/a |