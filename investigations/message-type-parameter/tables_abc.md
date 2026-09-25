
## micro_abc.tsv  (median of per-round minima; spread = max/min of the minima; allocs from round 1)

| case | A | B | C | D | E | B/A | C/A | D/A | E/A | spread A,C | allocs A/B/C/D/E |
|---|---|---|---|---|---|---|---|---|---|---|---|
| map/direct  NMV->out m[μ]::NMV m[v]::PM (BP) | 780 ns | 1.53 µs | 484 ns | 495 ns | 212 ns | 1.96 | 0.62 | 0.63 | 0.27 | 1.032,1.063 | 16/26/18/13/15 |
| map/barrier NMV->out m[μ]::NMV m[v]::PM (BP) | 805 ns | 1.55 µs | 498 ns | 505 ns | 248 ns | 1.93 | 0.62 | 0.63 | 0.31 | 1.043,1.070 | 17/27/19/14/16 |
| resolve/barrier NMV->out m[μ]::NMV m[v]::PM (BP) | 27 ns | 674 ns | 132 ns | 26 ns | 132 ns | 25.13 | 4.94 | 0.96 | 4.93 | 1.051,1.026 | 2/6/2/2/2 |
| deferred/barrier NMV->out m[μ]::NMV m[v]::PM (BP) | 797 ns | 1.56 µs | 509 ns | 521 ns | 245 ns | 1.96 | 0.64 | 0.65 | 0.31 | 1.030,1.046 | 17/27/19/14/16 |
| map/direct  NMP->μ q[out]::PM q[τ]::Gamma (VMP) | 898 ns | 1.83 µs | 435 ns | 630 ns | 154 ns | 2.04 | 0.48 | 0.70 | 0.17 | 1.035,1.018 | 14/21/14/11/11 |
| map/barrier NMP->μ q[out]::PM q[τ]::Gamma (VMP) | 931 ns | 1.85 µs | 456 ns | 648 ns | 196 ns | 1.99 | 0.49 | 0.70 | 0.21 | 1.045,1.048 | 15/22/15/12/12 |
| resolve/barrier NMP->μ q[out]::PM q[τ]::Gamma (VMP) | 27 ns | 1.00 µs | 136 ns | 26 ns | 134 ns | 37.61 | 5.09 | 0.97 | 5.01 | 1.055,1.031 | 2/6/2/2/2 |
| deferred/barrier NMP->μ q[out]::PM q[τ]::Gamma (VMP) | 934 ns | 1.86 µs | 454 ns | 646 ns | 198 ns | 1.99 | 0.49 | 0.69 | 0.21 | 1.041,1.016 | 15/22/15/12/12 |
| map/direct  MvNMC->out m[μ]::MvNMC(3) m[Σ]::PM (BP) | 819 ns | 1.65 µs | 526 ns | 512 ns | 249 ns | 2.01 | 0.64 | 0.62 | 0.30 | 1.049,1.091 | 18/28/20/15/17 |
| map/barrier MvNMC->out m[μ]::MvNMC(3) m[Σ]::PM (BP) | 827 ns | 1.68 µs | 552 ns | 536 ns | 271 ns | 2.04 | 0.67 | 0.65 | 0.33 | 1.038,1.072 | 19/29/21/16/18 |
| resolve/barrier MvNMC->out m[μ]::MvNMC(3) m[Σ]::PM (BP) | 26 ns | 727 ns | 186 ns | 26 ns | 182 ns | 27.65 | 7.06 | 0.97 | 6.91 | 1.056,1.002 | 2/6/2/2/2 |
| deferred/barrier MvNMC->out m[μ]::MvNMC(3) m[Σ]::PM (BP) | 841 ns | 1.70 µs | 557 ns | 547 ns | 274 ns | 2.02 | 0.66 | 0.65 | 0.33 | 1.045,1.076 | 19/29/21/16/18 |
| map/direct  Bernoulli->out m[p]::Beta | 929 ns | 1.36 µs | 481 ns | 661 ns | 200 ns | 1.47 | 0.52 | 0.71 | 0.21 | 1.043,1.054 | 18/23/18/15/15 |
| map/barrier Bernoulli->out m[p]::Beta | 968 ns | 1.38 µs | 501 ns | 683 ns | 230 ns | 1.42 | 0.52 | 0.71 | 0.24 | 1.034,1.050 | 19/24/19/16/16 |
| resolve/barrier Bernoulli->out m[p]::Beta | 27 ns | 712 ns | 128 ns | 26 ns | 127 ns | 26.57 | 4.76 | 0.96 | 4.74 | 1.063,1.023 | 2/6/2/2/2 |
| deferred/barrier Bernoulli->out m[p]::Beta | 972 ns | 1.40 µs | 507 ns | 685 ns | 238 ns | 1.44 | 0.52 | 0.70 | 0.25 | 1.026,1.027 | 19/24/19/16/16 |
| map/direct  Categorical->p q[out]::Categorical | 1.06 µs | 1.55 µs | 618 ns | 754 ns | 306 ns | 1.46 | 0.58 | 0.71 | 0.29 | 1.044,1.062 | 21/26/21/18/18 |
| map/barrier Categorical->p q[out]::Categorical | 1.07 µs | 1.56 µs | 639 ns | 781 ns | 326 ns | 1.45 | 0.59 | 0.73 | 0.30 | 1.012,1.042 | 22/27/22/19/19 |
| resolve/barrier Categorical->p q[out]::Categorical | 27 ns | 806 ns | 317 ns | 26 ns | 271 ns | 30.21 | 11.88 | 0.96 | 10.15 | 1.076,1.071 | 2/7/4/2/4 |
| deferred/barrier Categorical->p q[out]::Categorical | 1.08 µs | 1.58 µs | 645 ns | 790 ns | 336 ns | 1.46 | 0.60 | 0.73 | 0.31 | 1.077,1.035 | 22/27/22/19/19 |
| joint-marginal/barrier NMV (out,μ) | 148 ns | 1.26 µs | 236 ns | 151 ns | 232 ns | 8.53 | 1.60 | 1.03 | 1.57 | 1.022,1.017 | 11/19/16/11/16 |
| product/NMV x3 | 1.31 µs | 1.34 µs | 731 ns | 474 ns | 227 ns | 1.02 | 0.56 | 0.36 | 0.17 | 1.078,1.026 | 12/13/13/9/11 |
| product/deferred NMV x3 | 1.38 µs | 1.35 µs | 741 ns | 538 ns | 230 ns | 0.98 | 0.54 | 0.39 | 0.17 | 1.057,1.004 | 12/13/13/9/11 |
| product/NMV x10 | 1.50 µs | 3.98 µs | 1.40 µs | 660 ns | 954 ns | 2.65 | 0.93 | 0.44 | 0.63 | 1.102,1.009 | 33/48/48/30/46 |
| product/deferred NMV x10 | 1.67 µs | 3.96 µs | 1.42 µs | 838 ns | 929 ns | 2.37 | 0.85 | 0.50 | 0.56 | 1.076,1.032 | 33/48/48/30/46 |
| product/MvNMC(3) x3 | 2.13 µs | 2.09 µs | 1.57 µs | 1.30 µs | 1.04 µs | 0.98 | 0.74 | 0.61 | 0.49 | 1.054,1.046 | 34/35/35/31/33 |
| marginal-at-variable/NMV x3 | 1.58 µs | 1.36 µs | 820 ns | 528 ns | 270 ns | 0.86 | 0.52 | 0.33 | 0.17 | 1.048,1.065 | 14/14/14/10/12 |
| stream/marginal combineLatest x3 round | 111 ns | 107 ns | 111 ns | 110 ns | 108 ns | 0.97 | 1.00 | 0.99 | 0.97 | 1.046,1.015 | 0/0/0/0/0 |
| stream/message combineLatest x3 round | 109 ns | 113 ns | 109 ns | 108 ns | 112 ns | 1.04 | 1.00 | 0.99 | 1.03 | 1.059,1.038 | 0/0/0/0/0 |
| event/AfterMessageRuleCallEvent, result::Any | 266 ns | 265 ns | 272 ns | 268 ns | 276 ns | 0.99 | 1.02 | 1.01 | 1.04 | 1.024,1.029 | 3/3/3/3/3 |
| barrier/kernel(tuple, ann), Any-typed args | 24 ns | 23 ns | 24 ns | 23 ns | 24 ns | 0.94 | 1.01 | 0.95 | 1.00 | 1.080,1.043 | 1/1/1/1/1 |
| barrier/kernel(tuple, ann), known types | 4 ns | 4 ns | 4 ns | 4 ns | 4 ns | 1.00 | 1.00 | 1.00 | 1.00 | 1.029,1.029 | 0/0/0/0/0 |
| barrier/kernel(a, b), Any-typed args | 24 ns | 22 ns | 24 ns | 23 ns | 24 ns | 0.93 | 0.99 | 0.97 | 1.00 | 1.062,1.025 | 1/1/1/1/1 |
| unpack/map(getdata, 2 messages) | 2 ns | 29 ns | 29 ns | 2 ns | 29 ns | 14.40 | 14.35 | 1.00 | 14.35 | 1.000,1.021 | 0/1/1/0/1 |
| graph/iid n=1000 10 it | 12.19 ms | 44.07 ms | 14.62 ms | 9.14 ms | 12.57 ms | 3.62 | 1.20 | 0.75 | 1.03 | 1.018,1.059 | -1/-1/-1/-1/-1 |
| graph/chain n=300 10 it | 24.74 ms | 66.60 ms | 33.74 ms | 17.79 ms | 17.90 ms | 2.69 | 1.36 | 0.72 | 0.72 | 1.013,1.080 | -1/-1/-1/-1/-1 |

## model_abc.tsv  (median of per-round minima; spread = max/min of the minima; allocs from round 1)

| case | A | B | C | D | E | B/A | C/A | D/A | E/A | spread A,C | allocs A/B/C/D/E |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ssm1 load | 4779.0 ms | 4872.5 ms | 4680.5 ms | 4954.3 ms | 4767.3 ms | 1.02 | 0.98 | 1.04 | 1.00 | 1.049,1.058 | 0M/0M/0M/0M/0M |
| ssm1 ttfx | 11.19 s | 11.34 s | 11.20 s | 11.12 s | 11.24 s | 1.01 | 1.00 | 0.99 | 1.00 | 1.008,1.012 | 0M/0M/0M/0M/0M |
| ssm1 infer(I=1) | 111.5 ms | 129.0 ms | 115.9 ms | 109.6 ms | 112.8 ms | 1.16 | 1.04 | 0.98 | 1.01 | 1.031,1.010 | 78M/82M/80M/77M/78M |
| ssm2 load | 4938.2 ms | 5019.6 ms | 4995.6 ms | 4998.0 ms | 4817.8 ms | 1.02 | 1.01 | 1.01 | 0.98 | 1.045,1.005 | 0M/0M/0M/0M/0M |
| ssm2 ttfx | 14.90 s | 14.71 s | 14.69 s | 14.76 s | 14.55 s | 0.99 | 0.99 | 0.99 | 0.98 | 1.006,1.008 | 0M/0M/0M/0M/0M |
| ssm2 infer(I=1) | 192.1 ms | 218.1 ms | 196.6 ms | 186.3 ms | 186.6 ms | 1.14 | 1.02 | 0.97 | 0.97 | 1.009,1.031 | 130M/137M/133M/128M/130M |
| iid load | 4992.6 ms | 4721.3 ms | 4760.3 ms | 4756.1 ms | 4738.7 ms | 0.95 | 0.95 | 0.95 | 0.95 | 1.077,1.054 | 0M/0M/0M/0M/0M |
| iid ttfx | 9151.7 ms | 9201.5 ms | 9318.6 ms | 9307.1 ms | 9010.6 ms | 1.01 | 1.02 | 1.02 | 0.98 | 1.016,1.027 | 0M/0M/0M/0M/0M |
| iid infer(I=10) | 66.0 ms | 100.4 ms | 63.4 ms | 60.2 ms | 56.7 ms | 1.52 | 0.96 | 0.91 | 0.86 | 1.010,1.008 | 57M/70M/63M/52M/58M |
| iid infer(I=20) | 93.4 ms | 165.6 ms | 87.1 ms | 79.9 ms | 75.7 ms | 1.77 | 0.93 | 0.86 | 0.81 | 1.007,1.015 | 83M/110M/94M/73M/85M |
| iid per-iteration | 2.8 ms | 6.5 ms | 2.4 ms | 2.1 ms | 1.8 ms | 2.34 | 0.85 | 0.74 | 0.65 | 1.023,1.046 | 3M/4M/3M/2M/3M |
| nl load | 4738.3 ms | 4726.4 ms | 4739.8 ms | 4776.3 ms | 4958.1 ms | 1.00 | 1.00 | 1.01 | 1.05 | 1.073,1.186 | 0M/0M/0M/0M/0M |
| nl ttfx | 15.13 s | 14.78 s | 14.81 s | 15.05 s | 14.81 s | 0.98 | 0.98 | 1.00 | 0.98 | 1.210,1.014 | 0M/0M/0M/0M/0M |
| nl infer(I=5) | 83.4 ms | 138.2 ms | 93.4 ms | 74.3 ms | 77.9 ms | 1.66 | 1.12 | 0.89 | 0.93 | 1.055,1.015 | 54M/65M/59M/51M/54M |
| nl infer(I=10) | 111.4 ms | 218.2 ms | 131.0 ms | 96.9 ms | 102.9 ms | 1.96 | 1.18 | 0.87 | 0.92 | 1.061,1.026 | 72M/93M/83M/67M/71M |
| nl per-iteration | 5.8 ms | 16.7 ms | 7.5 ms | 4.5 ms | 5.0 ms | 2.89 | 1.30 | 0.77 | 0.87 | 1.112,1.134 | 4M/6M/5M/3M/4M |
| hmm load | 4992.7 ms | 4957.1 ms | 5012.5 ms | 4950.5 ms | 5024.3 ms | 0.99 | 1.00 | 0.99 | 1.01 | 1.112,1.093 | 0M/0M/0M/0M/0M |
| hmm ttfx | 15.66 s | 15.46 s | 15.32 s | 15.42 s | 15.24 s | 0.99 | 0.98 | 0.98 | 0.97 | 1.060,1.104 | 0M/0M/0M/0M/0M |
| hmm infer(I=10) | 88.0 ms | 136.3 ms | 94.2 ms | 77.0 ms | 78.2 ms | 1.55 | 1.07 | 0.88 | 0.89 | 1.029,1.014 | 72M/91M/79M/67M/71M |
| hmm infer(I=20) | 138.1 ms | 233.2 ms | 149.4 ms | 121.3 ms | 116.6 ms | 1.69 | 1.08 | 0.88 | 0.84 | 1.007,1.021 | 117M/157M/131M/107M/116M |
| hmm per-iteration | 5.1 ms | 9.8 ms | 5.5 ms | 4.4 ms | 3.9 ms | 1.92 | 1.09 | 0.87 | 0.77 | 1.052,1.084 | 5M/7M/5M/4M/4M |

## spec_abc.tsv (median over 2 runs)

| | A | B | C | D | E | C/A |
|---|---|---|---|---|---|---|
| ssm1 compile s | 11.94 | 13.05 | 11.71 | 11.89 | 13.03 | 0.981 |
| ssm2 compile s | 8.54 | 8.58 | 8.37 | 8.50 | 10.02 | 0.979 |
| iid compile s | 3.60 | 3.74 | 3.33 | 3.58 | 3.98 | 0.924 |
| nl compile s | 6.44 | 6.06 | 6.60 | 6.47 | 7.65 | 1.025 |
| hmm compile s | 8.88 | 8.98 | 8.62 | 8.85 | 9.52 | 0.970 |
| ReactiveMP specialisations | 5849 | 4384 | 4487 | 4370 | 3749 | 0.767 |
| Rocket specialisations | 19581 | 18123 | 18123 | 19581 | 18123 | 0.926 |
| RxInfer specialisations | 1265 | 1251 | 1251 | 1265 | 1251 | 0.989 |
