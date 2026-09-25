
## micro_f.tsv  (median of per-round minima; spread = max/min of the minima; allocs from round 1)

| case | A | C | E | F | C/A | E/A | F/A | spread A,C | allocs A/C/E/F |
|---|---|---|---|---|---|---|---|---|---|
| map/direct  NMV->out m[μ]::NMV m[v]::PM (BP) | 793 ns | 499 ns | 221 ns | 178 ns | 0.63 | 0.28 | 0.22 | 1.012,1.048 | 16/18/15/12 |
| map/barrier NMV->out m[μ]::NMV m[v]::PM (BP) | 803 ns | 520 ns | 242 ns | 190 ns | 0.65 | 0.30 | 0.24 | 1.008,1.049 | 17/19/16/13 |
| resolve/barrier NMV->out m[μ]::NMV m[v]::PM (BP) | 26 ns | 135 ns | 131 ns | 26 ns | 5.25 | 5.11 | 1.03 | 1.051,1.032 | 2/2/2/2 |
| deferred/barrier NMV->out m[μ]::NMV m[v]::PM (BP) | 816 ns | 521 ns | 258 ns | 202 ns | 0.64 | 0.32 | 0.25 | 1.018,1.057 | 17/19/16/13 |
| map/direct  NMP->μ q[out]::PM q[τ]::Gamma (VMP) | 914 ns | 436 ns | 163 ns | 124 ns | 0.48 | 0.18 | 0.14 | 1.008,1.010 | 14/14/11/8 |
| map/barrier NMP->μ q[out]::PM q[τ]::Gamma (VMP) | 920 ns | 453 ns | 185 ns | 141 ns | 0.49 | 0.20 | 0.15 | 1.005,1.007 | 15/15/12/9 |
| resolve/barrier NMP->μ q[out]::PM q[τ]::Gamma (VMP) | 26 ns | 137 ns | 139 ns | 27 ns | 5.22 | 5.28 | 1.02 | 1.062,1.015 | 2/2/2/2 |
| deferred/barrier NMP->μ q[out]::PM q[τ]::Gamma (VMP) | 932 ns | 456 ns | 199 ns | 152 ns | 0.49 | 0.21 | 0.16 | 1.006,1.013 | 15/15/12/9 |
| map/direct  MvNMC->out m[μ]::MvNMC(3) m[Σ]::PM (BP) | 817 ns | 536 ns | 254 ns | 199 ns | 0.66 | 0.31 | 0.24 | 1.028,1.029 | 18/20/17/14 |
| map/barrier MvNMC->out m[μ]::MvNMC(3) m[Σ]::PM (BP) | 851 ns | 558 ns | 269 ns | 220 ns | 0.65 | 0.32 | 0.26 | 1.056,1.017 | 19/21/18/15 |
| resolve/barrier MvNMC->out m[μ]::MvNMC(3) m[Σ]::PM (BP) | 25 ns | 185 ns | 184 ns | 26 ns | 7.27 | 7.22 | 1.03 | 1.052,1.007 | 2/2/2/2 |
| deferred/barrier MvNMC->out m[μ]::MvNMC(3) m[Σ]::PM (BP) | 868 ns | 564 ns | 278 ns | 229 ns | 0.65 | 0.32 | 0.26 | 1.047,1.027 | 19/21/18/15 |
| map/direct  Bernoulli->out m[p]::Beta | 946 ns | 487 ns | 206 ns | 174 ns | 0.51 | 0.22 | 0.18 | 1.023,1.037 | 18/18/15/12 |
| map/barrier Bernoulli->out m[p]::Beta | 964 ns | 507 ns | 233 ns | 194 ns | 0.53 | 0.24 | 0.20 | 1.035,1.030 | 19/19/16/13 |
| resolve/barrier Bernoulli->out m[p]::Beta | 26 ns | 127 ns | 132 ns | 27 ns | 4.96 | 5.14 | 1.04 | 1.039,1.005 | 2/2/2/2 |
| deferred/barrier Bernoulli->out m[p]::Beta | 977 ns | 514 ns | 262 ns | 202 ns | 0.53 | 0.27 | 0.21 | 1.029,1.035 | 19/19/16/13 |
| map/direct  Categorical->p q[out]::Categorical | 1.05 µs | 617 ns | 320 ns | 258 ns | 0.58 | 0.30 | 0.25 | 1.012,1.023 | 21/21/18/15 |
| map/barrier Categorical->p q[out]::Categorical | 1.07 µs | 642 ns | 325 ns | 286 ns | 0.60 | 0.30 | 0.27 | 1.023,1.013 | 22/22/19/16 |
| resolve/barrier Categorical->p q[out]::Categorical | 26 ns | 316 ns | 271 ns | 26 ns | 12.30 | 10.54 | 1.03 | 1.043,1.037 | 2/4/4/2 |
| deferred/barrier Categorical->p q[out]::Categorical | 1.08 µs | 674 ns | 353 ns | 298 ns | 0.62 | 0.33 | 0.28 | 1.027,1.011 | 22/22/19/16 |
| joint-marginal/barrier NMV (out,μ) | 150 ns | 235 ns | 236 ns | 153 ns | 1.57 | 1.57 | 1.02 | 1.023,1.020 | 11/16/16/11 |
| product/NMV x3 | 1.26 µs | 737 ns | 226 ns | 166 ns | 0.58 | 0.18 | 0.13 | 1.013,1.008 | 12/13/11/8 |
| product/deferred NMV x3 | 1.31 µs | 741 ns | 229 ns | 225 ns | 0.57 | 0.17 | 0.17 | 1.003,1.046 | 12/13/11/8 |
| product/NMV x10 | 1.44 µs | 1.41 µs | 928 ns | 346 ns | 0.98 | 0.65 | 0.24 | 1.017,1.024 | 33/48/46/29 |
| product/deferred NMV x10 | 1.61 µs | 1.42 µs | 905 ns | 559 ns | 0.88 | 0.56 | 0.35 | 1.008,1.029 | 33/48/46/29 |
| product/MvNMC(3) x3 | 2.07 µs | 1.59 µs | 1.03 µs | 979 ns | 0.77 | 0.50 | 0.47 | 1.013,1.021 | 34/35/33/30 |
| marginal-at-variable/NMV x3 | 1.62 µs | 828 ns | 261 ns | 206 ns | 0.51 | 0.16 | 0.13 | 1.008,1.021 | 14/14/12/9 |
| stream/marginal combineLatest x3 round | 108 ns | 111 ns | 111 ns | 112 ns | 1.02 | 1.02 | 1.03 | 1.058,1.010 | 0/0/0/0 |
| stream/message combineLatest x3 round | 110 ns | 108 ns | 114 ns | 113 ns | 0.98 | 1.03 | 1.03 | 1.018,1.020 | 0/0/0/0 |
| event/AfterMessageRuleCallEvent, result::Any | 271 ns | 272 ns | 275 ns | 273 ns | 1.00 | 1.01 | 1.01 | 1.014,1.003 | 3/3/3/3 |
| barrier/kernel(tuple, ann), Any-typed args | 23 ns | 24 ns | 24 ns | 24 ns | 1.04 | 1.05 | 1.06 | 1.062,1.013 | 1/1/1/1 |
| barrier/kernel(tuple, ann), known types | 4 ns | 4 ns | 4 ns | 3 ns | 1.00 | 1.00 | 0.97 | 1.000,1.029 | 0/0/0/0 |
| barrier/kernel(a, b), Any-typed args | 23 ns | 24 ns | 24 ns | 24 ns | 1.03 | 1.05 | 1.03 | 1.062,1.017 | 1/1/1/1 |
| unpack/map(getdata, 2 messages) | 2 ns | 29 ns | 29 ns | 2 ns | 14.35 | 14.35 | 1.00 | 1.000,1.000 | 0/1/1/0 |
| graph/iid n=1000 10 it | 12.05 ms | 15.52 ms | 11.90 ms | 5.75 ms | 1.29 | 0.99 | 0.48 | 1.040,1.088 | -1/-1/-1/-1 |
| graph/chain n=300 10 it | 24.41 ms | 32.97 ms | 17.91 ms | 12.58 ms | 1.35 | 0.73 | 0.52 | 1.015,1.053 | -1/-1/-1/-1 |

## model_f.tsv  (median of per-round minima; spread = max/min of the minima; allocs from round 1)

| case | A | C | E | F | C/A | E/A | F/A | spread A,C | allocs A/C/E/F |
|---|---|---|---|---|---|---|---|---|---|
| ssm1 load | 4986.6 ms | 4934.8 ms | 4985.2 ms | 5028.8 ms | 0.99 | 1.00 | 1.01 | 1.040,1.018 | 0M/0M/0M/0M |
| ssm1 ttfx | 11.37 s | 11.45 s | 11.42 s | 11.19 s | 1.01 | 1.00 | 0.98 | 1.046,1.038 | 0M/0M/0M/0M |
| ssm1 infer(I=1) | 112.7 ms | 116.6 ms | 111.0 ms | 108.7 ms | 1.03 | 0.98 | 0.96 | 1.043,1.030 | 78M/80M/78M/76M |
| ssm2 load | 5011.2 ms | 5036.8 ms | 4992.0 ms | 4975.7 ms | 1.01 | 1.00 | 0.99 | 1.010,1.014 | 0M/0M/0M/0M |
| ssm2 ttfx | 15.03 s | 15.06 s | 14.89 s | 14.97 s | 1.00 | 0.99 | 1.00 | 1.039,1.036 | 0M/0M/0M/0M |
| ssm2 infer(I=1) | 192.7 ms | 196.8 ms | 185.9 ms | 185.1 ms | 1.02 | 0.96 | 0.96 | 1.026,1.022 | 130M/133M/130M/128M |
| iid load | 5000.9 ms | 4987.1 ms | 4993.1 ms | 4937.4 ms | 1.00 | 1.00 | 0.99 | 1.077,1.007 | 0M/0M/0M/0M |
| iid ttfx | 9394.6 ms | 9278.5 ms | 9373.1 ms | 9127.1 ms | 0.99 | 1.00 | 0.97 | 1.052,1.032 | 0M/0M/0M/0M |
| iid infer(I=10) | 65.5 ms | 62.8 ms | 56.4 ms | 49.5 ms | 0.96 | 0.86 | 0.76 | 1.010,1.020 | 57M/63M/58M/49M |
| iid infer(I=20) | 92.5 ms | 86.9 ms | 74.7 ms | 59.1 ms | 0.94 | 0.81 | 0.64 | 1.010,1.017 | 83M/94M/85M/67M |
| iid per-iteration | 2.7 ms | 2.4 ms | 1.8 ms | 1.0 ms | 0.87 | 0.68 | 0.37 | 1.056,1.036 | 3M/3M/3M/2M |
| nl load | 5092.4 ms | 4977.1 ms | 4979.5 ms | 4996.2 ms | 0.98 | 0.98 | 0.98 | 1.025,1.029 | 0M/0M/0M/0M |
| nl ttfx | 15.10 s | 15.40 s | 14.74 s | 14.99 s | 1.02 | 0.98 | 0.99 | 1.033,1.042 | 0M/0M/0M/0M |
| nl infer(I=5) | 84.0 ms | 93.9 ms | 77.8 ms | 69.8 ms | 1.12 | 0.93 | 0.83 | 1.020,1.010 | 54M/59M/54M/50M |
| nl infer(I=10) | 110.7 ms | 131.5 ms | 102.1 ms | 86.1 ms | 1.19 | 0.92 | 0.78 | 1.022,1.014 | 72M/83M/71M/65M |
| nl per-iteration | 5.2 ms | 7.4 ms | 4.8 ms | 3.3 ms | 1.41 | 0.91 | 0.63 | 1.140,1.039 | 4M/5M/4M/3M |
| hmm load | 5009.3 ms | 5008.5 ms | 5031.8 ms | 5038.5 ms | 1.00 | 1.00 | 1.01 | 1.013,1.007 | 0M/0M/0M/0M |
| hmm ttfx | 15.71 s | 15.66 s | 15.61 s | 16.03 s | 1.00 | 0.99 | 1.02 | 1.025,1.030 | 0M/0M/0M/0M |
| hmm infer(I=10) | 90.5 ms | 98.2 ms | 76.8 ms | 70.3 ms | 1.09 | 0.85 | 0.78 | 1.069,1.057 | 72M/79M/71M/66M |
| hmm infer(I=20) | 140.5 ms | 154.4 ms | 116.8 ms | 107.2 ms | 1.10 | 0.83 | 0.76 | 1.006,1.053 | 117M/131M/116M/106M |
| hmm per-iteration | 5.0 ms | 5.8 ms | 3.9 ms | 3.8 ms | 1.17 | 0.78 | 0.75 | 1.106,1.088 | 5M/5M/4M/4M |

## spec_f.tsv (median over 2 runs)

| | A | C | E | F | C/A |
|---|---|---|---|---|---|
| ssm1 compile s | 11.53 | 11.76 | 11.55 | 11.43 | 1.019 |
| ssm2 compile s | 8.26 | 8.15 | 7.98 | 8.11 | 0.987 |
| iid compile s | 3.45 | 3.51 | 3.42 | 3.52 | 1.018 |
| nl compile s | 6.36 | 6.45 | 6.39 | 6.49 | 1.014 |
| hmm compile s | 9.19 | 9.01 | 8.84 | 9.14 | 0.981 |
| ReactiveMP specialisations | 5849 | 4487 | 3749 | 4396 | 0.767 |
| Rocket specialisations | 19581 | 18123 | 18123 | 19581 | 0.926 |
| RxInfer specialisations | 1265 | 1251 | 1251 | 1265 | 0.989 |
