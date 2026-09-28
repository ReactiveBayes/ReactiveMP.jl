| model | stage | v6 | A | P12 | E4 | FULL | FULLPC |
|---|---|---|---|---|---|---|---|
| betabern | infer(I=1) | 70.4 ms | 158.5 ms | 155.1 ms | 103.5 ms | 102.5 ms | 96.6 ms |
| betabern | ttfx | 6.71 s | 7.96 s | 7.92 s | 8.00 s | 7.86 s | 530.2 ms |
| betabern@50000 | infer(I=1) | 898.3 ms | 1.78 s | 1.74 s | 1.21 s | 1.18 s | 1.11 s |
| betabern@50000 | ttfx | 8.27 s | 10.41 s | 10.31 s | 9.96 s | 9.64 s | 2.04 s |
| filter | infer(I=1) | 9.1 ms | 13.6 ms | 7.2 ms | 6.1 ms | 5.6 ms | 4.6 ms |
| filter | ttfx | 7.36 s | 9.12 s | 9.07 s | 9.39 s | 9.48 s | 2.60 s |
| gmm | infer(I=10) | 52.6 ms | 148.1 ms | 113.4 ms | 77.0 ms | 74.4 ms | 70.6 ms |
| gmm | infer(I=20) | 80.8 ms | 222.5 ms | 146.2 ms | 106.7 ms | 101.7 ms | 97.8 ms |
| gmm | per-iteration | 2.9 ms | 7.4 ms | 3.3 ms | 3.0 ms | 2.8 ms | 2.7 ms |
| gmm | setup (T(I) - I*per-iteration) | 23.6 ms | 75.5 ms | 80.7 ms | 47.4 ms | 46.8 ms | 43.7 ms |
| gmm | ttfx | 10.49 s | 13.66 s | 13.46 s | 13.83 s | 13.76 s | 5.52 s |
| hmm | infer(I=10) | 46.5 ms | 86.7 ms | 69.5 ms | 55.1 ms | 54.1 ms | 53.3 ms |
| hmm | infer(I=20) | 76.8 ms | 138.0 ms | 102.2 ms | 87.1 ms | 87.1 ms | 85.2 ms |
| hmm | per-iteration | 3.0 ms | 5.1 ms | 3.3 ms | 3.2 ms | 3.3 ms | 3.2 ms |
| hmm | setup (T(I) - I*per-iteration) | 16.3 ms | 35.7 ms | 36.7 ms | 22.1 ms | 21.1 ms | 20.9 ms |
| hmm | ttfx | 14.20 s | 17.13 s | 16.95 s | 17.39 s | 17.54 s | 12.34 s |
| iid | infer(I=10) | 25.2 ms | 70.8 ms | 51.4 ms | 34.8 ms | 34.4 ms | 30.8 ms |
| iid | infer(I=20) | 31.8 ms | 100.7 ms | 63.0 ms | 43.1 ms | 42.0 ms | 37.5 ms |
| iid | per-iteration | 666 µs | 3.1 ms | 1.1 ms | 826 µs | 764 µs | 655 µs |
| iid | setup (T(I) - I*per-iteration) | 18.5 ms | 41.1 ms | 40.5 ms | 26.7 ms | 26.6 ms | 24.2 ms |
| iid | ttfx | 7.52 s | 8.71 s | 8.68 s | 8.90 s | 8.83 s | 684.4 ms |
| iid+defaults | infer(I=10) | 25.1 ms | 69.8 ms | 51.9 ms | 34.9 ms | 34.8 ms | 31.4 ms |
| iid+defaults | infer(I=20) | 31.5 ms | 99.8 ms | 63.5 ms | 43.7 ms | 43.1 ms | 38.1 ms |
| iid+defaults | per-iteration | 668 µs | 3.0 ms | 1.2 ms | 878 µs | 854 µs | 686 µs |
| iid+defaults | setup (T(I) - I*per-iteration) | 18.4 ms | 40.1 ms | 40.3 ms | 26.1 ms | 26.5 ms | 24.5 ms |
| iid+defaults | ttfx | 7.47 s | 8.75 s | 8.65 s | 8.87 s | 8.49 s | 2.24 s |
| iid@100 | infer(I=10) | 3.0 ms | 7.6 ms | 5.7 ms | 4.0 ms | 3.9 ms | 3.4 ms |
| iid@100 | infer(I=20) | 3.8 ms | 10.8 ms | 7.0 ms | 5.0 ms | 4.9 ms | 4.2 ms |
| iid@100 | per-iteration | 84 µs | 316 µs | 129 µs | 102 µs | 94 µs | 78 µs |
| iid@100 | setup (T(I) - I*per-iteration) | 2.1 ms | 4.5 ms | 4.4 ms | 3.0 ms | 3.0 ms | 2.7 ms |
| iid@100 | ttfx | 7.50 s | 8.69 s | 8.61 s | 8.84 s | 8.79 s | 640.0 ms |
| iid@10000 | infer(I=10) | 295.3 ms | 880.6 ms | 686.5 ms | 450.0 ms | 394.6 ms | 401.9 ms |
| iid@10000 | infer(I=20) | 408.7 ms | 1.36 s | 960.1 ms | 647.5 ms | 632.3 ms | 628.6 ms |
| iid@10000 | per-iteration | 11.2 ms | 48.3 ms | 25.0 ms | 19.2 ms | 23.8 ms | 22.7 ms |
| iid@10000 | setup (T(I) - I*per-iteration) | 180.3 ms | 406.3 ms | 448.4 ms | 268.7 ms | 156.9 ms | 175.2 ms |
| iid@10000 | ttfx | 8.03 s | 9.83 s | 9.58 s | 9.50 s | 9.44 s | 1.25 s |
| iid@100000 | infer(I=10) | 5.36 s | 11.79 s | 9.09 s | 7.39 s | 5.96 s | 6.11 s |
| iid@100000 | infer(I=20) | 7.72 s | 19.64 s | 14.12 s | 10.33 s | 8.94 s | 9.21 s |
| iid@100000 | per-iteration | 236.4 ms | 784.8 ms | 506.6 ms | 295.6 ms | 296.5 ms | 310.3 ms |
| iid@100000 | setup (T(I) - I*per-iteration) | 2.99 s | 3.94 s | 3.90 s | 4.45 s | 3.00 s | 3.01 s |
| iid@100000 | ttfx | 14.68 s | 22.91 s | 21.30 s | 18.73 s | 17.96 s | 9.53 s |
| linreg | infer(I=10) | — | 350.5 ms | 263.7 ms | 204.3 ms | 193.6 ms | 188.7 ms |
| linreg | infer(I=20) | — | 565.8 ms | 386.3 ms | 310.7 ms | 286.7 ms | 279.1 ms |
| linreg | per-iteration | — | 20.7 ms | 12.1 ms | 10.6 ms | 9.3 ms | 8.9 ms |
| linreg | setup (T(I) - I*per-iteration) | — | 139.8 ms | 138.4 ms | 98.3 ms | 100.5 ms | 100.8 ms |
| linreg | ttfx | — | 12.17 s | 12.07 s | 12.40 s | 14.78 s | 7.89 s |
| nl | infer(I=10) | 61.4 ms | 110.2 ms | 81.7 ms | 61.8 ms | 60.7 ms | 54.7 ms |
| nl | infer(I=5) | 43.5 ms | 81.6 ms | 67.7 ms | 49.4 ms | 48.6 ms | 45.7 ms |
| nl | per-iteration | 3.6 ms | 5.5 ms | 2.8 ms | 2.5 ms | 2.3 ms | 1.9 ms |
| nl | setup (T(I) - I*per-iteration) | 25.3 ms | 54.7 ms | 53.6 ms | 36.7 ms | 37.1 ms | 36.4 ms |
| nl | ttfx | 12.66 s | 16.23 s | 16.19 s | 16.53 s | 17.12 s | 10.09 s |
| ssm1 | infer(I=1) | 61.8 ms | 119.8 ms | 114.8 ms | 78.1 ms | 78.3 ms | 73.8 ms |
| ssm1 | ttfx | 11.46 s | 13.58 s | 13.51 s | 13.78 s | 13.65 s | 6.56 s |
| ssm1+defaults | infer(I=1) | 60.9 ms | 120.9 ms | 115.7 ms | 78.4 ms | 78.3 ms | 75.1 ms |
| ssm1+defaults | ttfx | 11.34 s | 13.50 s | 13.45 s | 13.62 s | 13.39 s | 6.97 s |
| ssm1@100 | infer(I=1) | 6.0 ms | 11.8 ms | 11.3 ms | 7.6 ms | 7.7 ms | 7.2 ms |
| ssm1@100 | ttfx | 9.10 s | 10.99 s | 10.97 s | 11.25 s | 13.01 s | 5.92 s |
| ssm1@10000 | infer(I=1) | 758.2 ms | 1.35 s | 1.30 s | 938.7 ms | 922.3 ms | 891.9 ms |
| ssm1@10000 | ttfx | 12.68 s | 15.61 s | 15.52 s | 15.30 s | 15.20 s | 7.98 s |
| ssm2 | infer(I=1) | 107.6 ms | 190.8 ms | 184.2 ms | 134.2 ms | 130.7 ms | 128.1 ms |
| ssm2 | ttfx | 13.32 s | 16.56 s | 16.46 s | 16.70 s | 16.77 s | 11.16 s |

| micro case | A | P12 | E4 | FULL | FULLPC |
|---|---|---|---|---|---|
| map/direct  NMV->out m[μ]::NMV m[v]::PM (BP) | 873 ns / 15 | 181 ns / 11 | 69 ns / 5 | 69 ns / 5 | 69 ns / 5 |
| map/barrier NMV->out m[μ]::NMV m[v]::PM (BP) | 894 ns / 16 | 209 ns / 12 | 98 ns / 6 | 101 ns / 6 | 99 ns / 6 |
| map/callbacks=(;) NMV->out m[μ]::NMV m[v]::PM (BP) | 1662 ns / 16 | 182 ns / 11 | 71 ns / 5 | 71 ns / 5 | 71 ns / 5 |
| resolve/barrier NMV->out m[μ]::NMV m[v]::PM (BP) | 26 ns / 2 | 26 ns / 2 | 26 ns / 2 | 26 ns / 2 | 26 ns / 2 |
| exec/execute_rule NMV->out m[μ]::NMV m[v]::PM (BP) | 49 ns / 4 | 49 ns / 4 | 49 ns / 4 | 48 ns / 4 | 50 ns / 4 |
| deferred/barrier NMV->out m[μ]::NMV m[v]::PM (BP) | 906 ns / 16 | 222 ns / 12 | 106 ns / 6 | 105 ns / 6 | 106 ns / 6 |
| map/direct  NMP->μ q[out]::PM q[τ]::Gamma (VMP) | 1046 ns / 17 | 184 ns / 11 | 70 ns / 5 | 70 ns / 5 | 69 ns / 5 |
| map/barrier NMP->μ q[out]::PM q[τ]::Gamma (VMP) | 1071 ns / 18 | 212 ns / 12 | 101 ns / 6 | 99 ns / 6 | 100 ns / 6 |
| map/callbacks=(;) NMP->μ q[out]::PM q[τ]::Gamma (VMP) | 1883 ns / 18 | 186 ns / 11 | 70 ns / 5 | 70 ns / 5 | 72 ns / 5 |
| resolve/barrier NMP->μ q[out]::PM q[τ]::Gamma (VMP) | 26 ns / 2 | 25 ns / 2 | 26 ns / 2 | 26 ns / 2 | 26 ns / 2 |
| exec/execute_rule NMP->μ q[out]::PM q[τ]::Gamma (VMP) | 49 ns / 4 | 49 ns / 4 | 49 ns / 4 | 50 ns / 4 | 49 ns / 4 |
| deferred/barrier NMP->μ q[out]::PM q[τ]::Gamma (VMP) | 1083 ns / 18 | 223 ns / 12 | 107 ns / 6 | 107 ns / 6 | 107 ns / 6 |
| map/direct  NMP->τ q[out]::PM q[μ]::NMV (VMP) | 1042 ns / 17 | 196 ns / 11 | 71 ns / 5 | 71 ns / 5 | 69 ns / 5 |
| map/barrier NMP->τ q[out]::PM q[μ]::NMV (VMP) | 1050 ns / 18 | 217 ns / 12 | 98 ns / 6 | 98 ns / 6 | 98 ns / 6 |
| map/callbacks=(;) NMP->τ q[out]::PM q[μ]::NMV (VMP) | 1883 ns / 18 | 189 ns / 11 | 72 ns / 5 | 73 ns / 5 | 74 ns / 5 |
| resolve/barrier NMP->τ q[out]::PM q[μ]::NMV (VMP) | 26 ns / 2 | 26 ns / 2 | 26 ns / 2 | 26 ns / 2 | 26 ns / 2 |
| exec/execute_rule NMP->τ q[out]::PM q[μ]::NMV (VMP) | 49 ns / 4 | 50 ns / 4 | 50 ns / 4 | 51 ns / 4 | 50 ns / 4 |
| deferred/barrier NMP->τ q[out]::PM q[μ]::NMV (VMP) | 1062 ns / 18 | 220 ns / 12 | 105 ns / 6 | 103 ns / 6 | 105 ns / 6 |
| map/direct  MvNMC->out m[μ]::MvNMC(3) m[Σ]::PM (BP) | 903 ns / 17 | 211 ns / 13 | 98 ns / 7 | 97 ns / 7 | 98 ns / 7 |
| map/barrier MvNMC->out m[μ]::MvNMC(3) m[Σ]::PM (BP) | 917 ns / 18 | 234 ns / 14 | 119 ns / 8 | 121 ns / 8 | 119 ns / 8 |
| map/callbacks=(;) MvNMC->out m[μ]::MvNMC(3) m[Σ]::PM (BP) | 1717 ns / 18 | 213 ns / 13 | 98 ns / 7 | 100 ns / 7 | 99 ns / 7 |
| resolve/barrier MvNMC->out m[μ]::MvNMC(3) m[Σ]::PM (BP) | 26 ns / 2 | 25 ns / 2 | 26 ns / 2 | 26 ns / 2 | 25 ns / 2 |
| exec/execute_rule MvNMC->out m[μ]::MvNMC(3) m[Σ]::PM (BP) | 70 ns / 6 | 70 ns / 6 | 71 ns / 6 | 69 ns / 6 | 71 ns / 6 |
| deferred/barrier MvNMC->out m[μ]::MvNMC(3) m[Σ]::PM (BP) | 933 ns / 18 | 249 ns / 14 | 136 ns / 8 | 136 ns / 8 | 138 ns / 8 |
| map/direct  Bernoulli->out m[p]::Beta | 1033 ns / 17 | 180 ns / 11 | 68 ns / 5 | 68 ns / 5 | 68 ns / 5 |
| map/barrier Bernoulli->out m[p]::Beta | 1058 ns / 18 | 205 ns / 12 | 96 ns / 6 | 97 ns / 6 | 95 ns / 6 |
| map/callbacks=(;) Bernoulli->out m[p]::Beta | 1879 ns / 18 | 184 ns / 11 | 73 ns / 5 | 73 ns / 5 | 72 ns / 5 |
| resolve/barrier Bernoulli->out m[p]::Beta | 26 ns / 2 | 26 ns / 2 | 26 ns / 2 | 26 ns / 2 | 26 ns / 2 |
| exec/execute_rule Bernoulli->out m[p]::Beta | 51 ns / 4 | 49 ns / 4 | 49 ns / 4 | 51 ns / 4 | 50 ns / 4 |
| deferred/barrier Bernoulli->out m[p]::Beta | 1075 ns / 18 | 216 ns / 12 | 102 ns / 6 | 101 ns / 6 | 103 ns / 6 |
| map/direct  Categorical->p q[out]::Categorical | 1129 ns / 19 | 271 ns / 13 | 169 ns / 7 | 171 ns / 7 | 167 ns / 7 |
| map/barrier Categorical->p q[out]::Categorical | 1150 ns / 20 | 298 ns / 14 | 184 ns / 8 | 185 ns / 8 | 187 ns / 8 |
| map/callbacks=(;) Categorical->p q[out]::Categorical | 1967 ns / 20 | 278 ns / 13 | 171 ns / 7 | 170 ns / 7 | 170 ns / 7 |
| resolve/barrier Categorical->p q[out]::Categorical | 26 ns / 2 | 26 ns / 2 | 26 ns / 2 | 26 ns / 2 | 26 ns / 2 |
| exec/execute_rule Categorical->p q[out]::Categorical | 130 ns / 6 | 131 ns / 6 | 132 ns / 6 | 128 ns / 6 | 133 ns / 6 |
| deferred/barrier Categorical->p q[out]::Categorical | 1158 ns / 20 | 308 ns / 14 | 200 ns / 8 | 195 ns / 8 | 198 ns / 8 |
| joint-marginal/barrier NMV (out,μ) | 236 ns / 15 | 235 ns / 15 | 148 ns / 9 | 147 ns / 9 | 149 ns / 9 |
| product/NMV x2 | 1262 ns / 9 | 514 ns / 6 | 527 ns / 6 | 528 ns / 6 | 530 ns / 6 |
| product/NMV x5 | 1329 ns / 18 | 585 ns / 15 | 598 ns / 15 | 606 ns / 15 | 602 ns / 15 |
| product/NMV x10 | 1446 ns / 33 | 699 ns / 30 | 730 ns / 30 | 716 ns / 30 | 722 ns / 30 |
| product/NMV x100 | 3474 ns / 303 | 2639 ns / 300 | 2708 ns / 300 | 2704 ns / 300 | 2713 ns / 300 |
| product/deferred NMV x10 | 1642 ns / 33 | 920 ns / 30 | 909 ns / 30 | 925 ns / 30 | 930 ns / 30 |
| product/MvNMC(3) x3 | 2083 ns / 34 | 1362 ns / 31 | 1346 ns / 31 | 1371 ns / 31 | 1362 ns / 31 |
| product/Gamma x10 | 1471 ns / 33 | 715 ns / 30 | 734 ns / 30 | 733 ns / 30 | 721 ns / 30 |
| marginal-at-variable/NMV x3 | 1575 ns / 14 | 594 ns / 10 | 598 ns / 10 | 614 ns / 10 | 610 ns / 10 |
| rocket/subject fan-out k=1 | 16 ns / 0 | 15 ns / 0 | 16 ns / 0 | 16 ns / 0 | 16 ns / 0 |
| rocket/subject fan-out k=10 | 105 ns / 0 | 108 ns / 0 | 110 ns / 0 | 106 ns / 0 | 104 ns / 0 |
| rocket/subject fan-out k=100 | 962 ns / 0 | 930 ns / 0 | 978 ns / 0 | 951 ns / 0 | 927 ns / 0 |
| rocket/combineLatest PushNew k=2 round | 68 ns / 0 | 68 ns / 0 | 68 ns / 0 | 67 ns / 0 | 39 ns / 0 |
| rocket/combineLatest PushNew k=5 round | 159 ns / 0 | 160 ns / 0 | 161 ns / 0 | 159 ns / 0 | 90 ns / 0 |
| rocket/combineLatest PushNew k=10 round | 358 ns / 0 | 360 ns / 0 | 364 ns / 0 | 342 ns / 0 | 201 ns / 0 |
| rocket/combineLatest PushNew k=20 round | 716 ns / 0 | 730 ns / 0 | 718 ns / 0 | 680 ns / 0 | 384 ns / 0 |
| rocket/collectLatest N=100 round | 2083 ns / 100 | 2125 ns / 100 | 2125 ns / 100 | 1875 ns / 100 | 1834 ns / 100 |
| rocket/collectLatest N=1000 round | 22958 ns / 1000 | 22958 ns / 1000 | 23250 ns / 1000 | 19375 ns / 1000 | 19542 ns / 1000 |
| rocket/collectLatest N=10000 round | 484416 ns / 10000 | 485834 ns / 10000 | 482375 ns / 10000 | 194083 ns / 10000 | 196125 ns / 10000 |
| event/AfterMessageRuleCallEvent, result::Any | 298 ns / 3 | 296 ns / 3 | 302 ns / 3 | 295 ns / 3 | 306 ns / 3 |
| event/generate_span_id((;)) | 758 ns / 0 | 4 ns / 0 | 4 ns / 0 | 4 ns / 0 | 4 ns / 0 |
| ctor/Message from Any | 382 ns / 2 | 380 ns / 2 | 385 ns / 2 | 391 ns / 2 | 380 ns / 2 |
| ctor/AnnotationDict() | 4 ns / 1 | 4 ns / 1 | 4 ns / 1 | 4 ns / 1 | 4 ns / 1 |
| graph/iid n=100 per-iteration | 135908 ns / -1 | 59850 ns / -1 | 47525 ns / -1 | 44858 ns / -1 | 45900 ns / -1 |
| graph/iid n=100 build+activate | 2513333 ns / -1 | 2518333 ns / -1 | 1340208 ns / -1 | 1355708 ns / -1 | 1367042 ns / -1 |
| graph/iid n=1000 per-iteration | 1332450 ns / -1 | 597808 ns / -1 | 455962 ns / -1 | 436167 ns / -1 | 441062 ns / -1 |
| graph/iid n=1000 build+activate | 24065209 ns / -1 | 24425208 ns / -1 | 12284125 ns / -1 | 12574000 ns / -1 | 12647625 ns / -1 |
| graph/iid n=10000 per-iteration | 20293383 ns / -1 | 12322404 ns / -1 | 6791138 ns / -1 | 5592475 ns / -1 | 5625154 ns / -1 |
| graph/iid n=10000 build+activate | 247947542 ns / -1 | 254572958 ns / -1 | 129997958 ns / -1 | 132107125 ns / -1 | 132222959 ns / -1 |
| graph/iid n=1000 no-FE per-iteration | 1191796 ns / -1 | 457917 ns / -1 | 322629 ns / -1 | 324433 ns / -1 | 324000 ns / -1 |
| graph/iid n=1000 no-FE build+activate | 20476541 ns / -1 | 20898375 ns / -1 | 11280708 ns / -1 | 11326000 ns / -1 | 11384958 ns / -1 |
| graph/chain n=300 per-iteration | 2567942 ns / -1 | 1333383 ns / -1 | 1097975 ns / -1 | 1079758 ns / -1 | 1068662 ns / -1 |
| graph/chain n=300 build+activate | 21472583 ns / -1 | 21986125 ns / -1 | 10079250 ns / -1 | 11231583 ns / -1 | 10876209 ns / -1 |
| graph/chain n=1000 per-iteration | 9440292 ns / -1 | 5210050 ns / -1 | — | 4209433 ns / -1 | 4157158 ns / -1 |
| graph/chain n=1000 build+activate | 74794000 ns / -1 | 74881875 ns / -1 | — | 40298917 ns / -1 | 38501833 ns / -1 |
| graph/chain n=1000 no-FE per-iteration | 7455575 ns / -1 | 3810317 ns / -1 | — | 3129325 ns / -1 | 3089475 ns / -1 |
| graph/chain n=1000 no-FE build+activate | 62744333 ns / -1 | 62637667 ns / -1 | — | 35891208 ns / -1 | 34098125 ns / -1 |
