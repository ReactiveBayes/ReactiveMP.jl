| model | stage | v6 | A | P12 | E4 | FULL | FULLT | FULLPC | FULLTPC |
|---|---|---|---|---|---|---|---|---|---|
| betabern | infer(I=1) | 71.4 ms | — | — | — | 102.5 ms | 102.0 ms | — | 95.8 ms |
| betabern | ttfx | 6.69 s | — | — | — | 7.87 s | 7.83 s | — | 531.8 ms |
| betabern@50000 | infer(I=1) | 899.1 ms | — | — | — | 1.16 s | 1.16 s | — | 1.10 s |
| betabern@50000 | ttfx | 8.25 s | — | — | — | 9.57 s | 9.42 s | — | 2.05 s |
| filter | infer(I=1) | 9.0 ms | — | — | — | 5.9 ms | 5.3 ms | — | 4.4 ms |
| filter | ttfx | 7.33 s | — | — | — | 9.38 s | 9.54 s | — | 2.59 s |
| gmm | infer(I=10) | 51.9 ms | — | — | — | 74.0 ms | 70.6 ms | — | 66.8 ms |
| gmm | infer(I=20) | 81.9 ms | — | — | — | 100.3 ms | 94.0 ms | — | 91.7 ms |
| gmm | per-iteration | 3.0 ms | — | — | — | 2.7 ms | 2.4 ms | — | 2.3 ms |
| gmm | setup (T(I) - I*per-iteration) | 21.7 ms | — | — | — | 47.2 ms | 46.5 ms | — | 44.4 ms |
| gmm | ttfx | 10.45 s | — | — | — | 13.76 s | 13.65 s | — | 5.48 s |
| hmm | infer(I=10) | 45.6 ms | — | — | — | 54.2 ms | 50.2 ms | — | 49.8 ms |
| hmm | infer(I=20) | 76.8 ms | — | — | — | 85.7 ms | 79.1 ms | — | 78.1 ms |
| hmm | per-iteration | 3.1 ms | — | — | — | 3.1 ms | 3.0 ms | — | 2.9 ms |
| hmm | setup (T(I) - I*per-iteration) | 14.4 ms | — | — | — | 23.3 ms | 20.0 ms | — | 21.8 ms |
| hmm | ttfx | 14.07 s | — | — | — | 17.50 s | 17.67 s | — | 12.52 s |
| iid | infer(I=10) | 24.5 ms | — | — | — | 34.2 ms | 32.9 ms | — | 29.5 ms |
| iid | infer(I=20) | 31.0 ms | — | — | — | 42.7 ms | 39.9 ms | — | 35.6 ms |
| iid | per-iteration | 647 µs | — | — | — | 829 µs | 705 µs | — | 566 µs |
| iid | setup (T(I) - I*per-iteration) | 18.1 ms | — | — | — | 25.9 ms | 25.5 ms | — | 23.8 ms |
| iid | ttfx | 7.51 s | — | — | — | 8.74 s | 8.84 s | — | 686.8 ms |
| iid+defaults | infer(I=10) | 24.6 ms | — | — | — | 34.3 ms | 33.1 ms | — | 30.1 ms |
| iid+defaults | infer(I=20) | 31.2 ms | — | — | — | 42.9 ms | 40.8 ms | — | 35.4 ms |
| iid+defaults | per-iteration | 632 µs | — | — | — | 839 µs | 764 µs | — | 567 µs |
| iid+defaults | setup (T(I) - I*per-iteration) | 18.3 ms | — | — | — | 25.8 ms | 25.5 ms | — | 24.7 ms |
| iid+defaults | ttfx | 7.46 s | — | — | — | 8.43 s | 8.52 s | — | 2.23 s |
| iid@100 | infer(I=10) | 3.0 ms | — | — | — | 3.8 ms | 3.8 ms | — | 3.3 ms |
| iid@100 | infer(I=20) | 3.9 ms | — | — | — | 4.8 ms | 4.6 ms | — | 4.0 ms |
| iid@100 | per-iteration | 82 µs | — | — | — | 97 µs | 84 µs | — | 72 µs |
| iid@100 | setup (T(I) - I*per-iteration) | 2.1 ms | — | — | — | 2.9 ms | 2.9 ms | — | 2.6 ms |
| iid@100 | ttfx | 7.42 s | — | — | — | 8.70 s | 8.76 s | — | 633.6 ms |
| iid@10000 | infer(I=10) | 281.9 ms | — | — | — | 390.2 ms | 367.2 ms | — | 376.6 ms |
| iid@10000 | infer(I=20) | 449.9 ms | — | — | — | 627.7 ms | 510.4 ms | — | 494.3 ms |
| iid@10000 | per-iteration | 16.8 ms | — | — | — | 23.2 ms | 14.5 ms | — | 11.8 ms |
| iid@10000 | setup (T(I) - I*per-iteration) | 113.9 ms | — | — | — | 158.4 ms | 220.6 ms | — | 257.3 ms |
| iid@10000 | ttfx | 7.99 s | — | — | — | 9.37 s | 9.44 s | — | 1.20 s |
| iid@100000 | infer(I=10) | 5.29 s | — | — | — | 5.89 s | 5.36 s | — | 5.49 s |
| iid@100000 | infer(I=20) | 7.47 s | — | — | — | 8.99 s | 7.41 s | — | 7.24 s |
| iid@100000 | per-iteration | 217.8 ms | — | — | — | 309.9 ms | 204.4 ms | — | 175.7 ms |
| iid@100000 | setup (T(I) - I*per-iteration) | 3.12 s | — | — | — | 2.80 s | 3.32 s | — | 3.73 s |
| iid@100000 | ttfx | 14.58 s | — | — | — | 17.93 s | 16.57 s | — | 8.55 s |
| linreg | infer(I=10) | — | — | — | — | 194.3 ms | 185.9 ms | — | 179.8 ms |
| linreg | infer(I=20) | — | — | — | — | 286.6 ms | 266.3 ms | — | 257.9 ms |
| linreg | per-iteration | — | — | — | — | 9.0 ms | 7.9 ms | — | 7.9 ms |
| linreg | setup (T(I) - I*per-iteration) | — | — | — | — | 110.8 ms | 107.1 ms | — | 102.8 ms |
| linreg | ttfx | — | — | — | — | 14.72 s | 14.87 s | — | 7.98 s |
| nl | infer(I=10) | 61.8 ms | — | — | — | 60.0 ms | 58.4 ms | — | 51.6 ms |
| nl | infer(I=5) | 43.4 ms | — | — | — | 48.1 ms | 47.4 ms | — | 43.4 ms |
| nl | per-iteration | 3.7 ms | — | — | — | 2.4 ms | 2.3 ms | — | 1.9 ms |
| nl | setup (T(I) - I*per-iteration) | 24.9 ms | — | — | — | 36.1 ms | 35.6 ms | — | 34.2 ms |
| nl | ttfx | 12.60 s | — | — | — | 17.05 s | 16.95 s | — | 9.94 s |
| ssm1 | infer(I=1) | 59.0 ms | — | — | — | 78.8 ms | 78.0 ms | — | 74.7 ms |
| ssm1 | ttfx | 11.40 s | — | — | — | 13.72 s | 13.61 s | — | 6.59 s |
| ssm1+defaults | infer(I=1) | 60.7 ms | — | — | — | 78.5 ms | 77.3 ms | — | 74.0 ms |
| ssm1+defaults | ttfx | 11.30 s | — | — | — | 13.38 s | 13.36 s | — | 6.92 s |
| ssm1@100 | infer(I=1) | 6.0 ms | — | — | — | 7.7 ms | 7.7 ms | — | 7.1 ms |
| ssm1@100 | ttfx | 9.06 s | — | — | — | 12.99 s | 12.94 s | — | 5.95 s |
| ssm1@10000 | infer(I=1) | 725.1 ms | — | — | — | 915.8 ms | 914.1 ms | — | 885.5 ms |
| ssm1@10000 | ttfx | 12.62 s | — | — | — | 15.21 s | 15.17 s | — | 7.97 s |
| ssm2 | infer(I=1) | 106.3 ms | — | — | — | 131.0 ms | 132.9 ms | — | 128.1 ms |
| ssm2 | ttfx | 13.24 s | — | — | — | 16.59 s | 16.71 s | — | 11.23 s |

| micro case | A | P12 | E4 | FULL | FULLT | FULLPC | FULLTPC |
|---|---|---|---|---|---|---|---|
| map/direct  NMV->out m[μ]::NMV m[v]::PM (BP) | — | — | — | 67 ns / 5 | 27 ns / 2 | — | 28 ns / 2 |
| map/barrier NMV->out m[μ]::NMV m[v]::PM (BP) | — | — | — | 97 ns / 6 | 40 ns / 3 | — | 39 ns / 3 |
| map/callbacks=(;) NMV->out m[μ]::NMV m[v]::PM (BP) | — | — | — | 71 ns / 5 | 29 ns / 2 | — | 29 ns / 2 |
| resolve/barrier NMV->out m[μ]::NMV m[v]::PM (BP) | — | — | — | 25 ns / 2 | 24 ns / 2 | — | 24 ns / 2 |
| exec/execute_rule NMV->out m[μ]::NMV m[v]::PM (BP) | — | — | — | 48 ns / 4 | 4 ns / 0 | — | 4 ns / 0 |
| deferred/barrier NMV->out m[μ]::NMV m[v]::PM (BP) | — | — | — | 104 ns / 6 | 40 ns / 3 | — | 41 ns / 3 |
| map/direct  NMP->μ q[out]::PM q[τ]::Gamma (VMP) | — | — | — | 68 ns / 5 | 26 ns / 2 | — | 28 ns / 2 |
| map/barrier NMP->μ q[out]::PM q[τ]::Gamma (VMP) | — | — | — | 99 ns / 6 | 39 ns / 3 | — | 39 ns / 3 |
| map/callbacks=(;) NMP->μ q[out]::PM q[τ]::Gamma (VMP) | — | — | — | 70 ns / 5 | 28 ns / 2 | — | 28 ns / 2 |
| resolve/barrier NMP->μ q[out]::PM q[τ]::Gamma (VMP) | — | — | — | 26 ns / 2 | 24 ns / 2 | — | 24 ns / 2 |
| exec/execute_rule NMP->μ q[out]::PM q[τ]::Gamma (VMP) | — | — | — | 49 ns / 4 | 4 ns / 0 | — | 4 ns / 0 |
| deferred/barrier NMP->μ q[out]::PM q[τ]::Gamma (VMP) | — | — | — | 106 ns / 6 | 41 ns / 3 | — | 40 ns / 3 |
| map/direct  NMP->τ q[out]::PM q[μ]::NMV (VMP) | — | — | — | 70 ns / 5 | 27 ns / 2 | — | 27 ns / 2 |
| map/barrier NMP->τ q[out]::PM q[μ]::NMV (VMP) | — | — | — | 99 ns / 6 | 39 ns / 3 | — | 40 ns / 3 |
| map/callbacks=(;) NMP->τ q[out]::PM q[μ]::NMV (VMP) | — | — | — | 70 ns / 5 | 28 ns / 2 | — | 29 ns / 2 |
| resolve/barrier NMP->τ q[out]::PM q[μ]::NMV (VMP) | — | — | — | 26 ns / 2 | 24 ns / 2 | — | 24 ns / 2 |
| exec/execute_rule NMP->τ q[out]::PM q[μ]::NMV (VMP) | — | — | — | 50 ns / 4 | 4 ns / 0 | — | 4 ns / 0 |
| deferred/barrier NMP->τ q[out]::PM q[μ]::NMV (VMP) | — | — | — | 102 ns / 6 | 42 ns / 3 | — | 42 ns / 3 |
| map/direct  MvNMC->out m[μ]::MvNMC(3) m[Σ]::PM (BP) | — | — | — | 96 ns / 7 | 41 ns / 4 | — | 41 ns / 4 |
| map/barrier MvNMC->out m[μ]::MvNMC(3) m[Σ]::PM (BP) | — | — | — | 118 ns / 8 | 59 ns / 5 | — | 58 ns / 5 |
| map/callbacks=(;) MvNMC->out m[μ]::MvNMC(3) m[Σ]::PM (BP) | — | — | — | 95 ns / 7 | 42 ns / 4 | — | 42 ns / 4 |
| resolve/barrier MvNMC->out m[μ]::MvNMC(3) m[Σ]::PM (BP) | — | — | — | 25 ns / 2 | 24 ns / 2 | — | 24 ns / 2 |
| exec/execute_rule MvNMC->out m[μ]::MvNMC(3) m[Σ]::PM (BP) | — | — | — | 70 ns / 6 | 21 ns / 2 | — | 21 ns / 2 |
| deferred/barrier MvNMC->out m[μ]::MvNMC(3) m[Σ]::PM (BP) | — | — | — | 134 ns / 8 | 84 ns / 5 | — | 82 ns / 5 |
| map/direct  Bernoulli->out m[p]::Beta | — | — | — | 68 ns / 5 | 27 ns / 2 | — | 27 ns / 2 |
| map/barrier Bernoulli->out m[p]::Beta | — | — | — | 95 ns / 6 | 39 ns / 3 | — | 39 ns / 3 |
| map/callbacks=(;) Bernoulli->out m[p]::Beta | — | — | — | 71 ns / 5 | 28 ns / 2 | — | 27 ns / 2 |
| resolve/barrier Bernoulli->out m[p]::Beta | — | — | — | 26 ns / 2 | 24 ns / 2 | — | 24 ns / 2 |
| exec/execute_rule Bernoulli->out m[p]::Beta | — | — | — | 49 ns / 4 | 4 ns / 0 | — | 4 ns / 0 |
| deferred/barrier Bernoulli->out m[p]::Beta | — | — | — | 100 ns / 6 | 42 ns / 3 | — | 41 ns / 3 |
| map/direct  Categorical->p q[out]::Categorical | — | — | — | 165 ns / 7 | 126 ns / 4 | — | 126 ns / 4 |
| map/barrier Categorical->p q[out]::Categorical | — | — | — | 181 ns / 8 | 136 ns / 5 | — | 133 ns / 5 |
| map/callbacks=(;) Categorical->p q[out]::Categorical | — | — | — | 168 ns / 7 | 128 ns / 4 | — | 126 ns / 4 |
| resolve/barrier Categorical->p q[out]::Categorical | — | — | — | 26 ns / 2 | 24 ns / 2 | — | 24 ns / 2 |
| exec/execute_rule Categorical->p q[out]::Categorical | — | — | — | 131 ns / 6 | 55 ns / 2 | — | 55 ns / 2 |
| deferred/barrier Categorical->p q[out]::Categorical | — | — | — | 196 ns / 8 | 166 ns / 5 | — | 155 ns / 5 |
| joint-marginal/barrier NMV (out,μ) | — | — | — | 147 ns / 9 | 97 ns / 7 | — | 95 ns / 7 |
| product/NMV x2 | — | — | — | 524 ns / 6 | 529 ns / 6 | — | 527 ns / 6 |
| product/NMV x5 | — | — | — | 601 ns / 15 | 600 ns / 15 | — | 600 ns / 15 |
| product/NMV x10 | — | — | — | 711 ns / 30 | 708 ns / 30 | — | 713 ns / 30 |
| product/NMV x100 | — | — | — | 2685 ns / 300 | 2681 ns / 300 | — | 2648 ns / 300 |
| product/deferred NMV x10 | — | — | — | 928 ns / 30 | 908 ns / 30 | — | 912 ns / 30 |
| product/MvNMC(3) x3 | — | — | — | 1342 ns / 31 | 1367 ns / 31 | — | 1342 ns / 31 |
| product/Gamma x10 | — | — | — | 718 ns / 30 | 725 ns / 30 | — | 723 ns / 30 |
| marginal-at-variable/NMV x3 | — | — | — | 606 ns / 10 | 599 ns / 10 | — | 603 ns / 10 |
| rocket/subject fan-out k=1 | — | — | — | 16 ns / 0 | 16 ns / 0 | — | 16 ns / 0 |
| rocket/subject fan-out k=10 | — | — | — | 104 ns / 0 | 106 ns / 0 | — | 104 ns / 0 |
| rocket/subject fan-out k=100 | — | — | — | 926 ns / 0 | 927 ns / 0 | — | 950 ns / 0 |
| rocket/combineLatest PushNew k=2 round | — | — | — | 68 ns / 0 | 39 ns / 0 | — | 39 ns / 0 |
| rocket/combineLatest PushNew k=5 round | — | — | — | 160 ns / 0 | 91 ns / 0 | — | 90 ns / 0 |
| rocket/combineLatest PushNew k=10 round | — | — | — | 341 ns / 0 | 200 ns / 0 | — | 198 ns / 0 |
| rocket/combineLatest PushNew k=20 round | — | — | — | 677 ns / 0 | 397 ns / 0 | — | 388 ns / 0 |
| rocket/collectLatest N=100 round | — | — | — | 1875 ns / 100 | 1875 ns / 100 | — | 1833 ns / 100 |
| rocket/collectLatest N=1000 round | — | — | — | 19708 ns / 1000 | 19208 ns / 1000 | — | 19416 ns / 1000 |
| rocket/collectLatest N=10000 round | — | — | — | 197500 ns / 10000 | 193917 ns / 10000 | — | 196750 ns / 10000 |
| event/AfterMessageRuleCallEvent, result::Any | — | — | — | 295 ns / 3 | 294 ns / 3 | — | 302 ns / 3 |
| event/generate_span_id((;)) | — | — | — | 4 ns / 0 | 4 ns / 0 | — | 4 ns / 0 |
| ctor/Message from Any | — | — | — | 388 ns / 2 | 383 ns / 2 | — | 386 ns / 2 |
| ctor/AnnotationDict() | — | — | — | 4 ns / 1 | 4 ns / 1 | — | 4 ns / 1 |
| graph/iid n=100 per-iteration | — | — | — | 46883 ns / -1 | 40562 ns / -1 | — | 39362 ns / -1 |
| graph/iid n=100 build+activate | — | — | — | 1354042 ns / -1 | 1362125 ns / -1 | — | 1354334 ns / -1 |
| graph/iid n=1000 per-iteration | — | — | — | 452142 ns / -1 | 373771 ns / -1 | — | 371233 ns / -1 |
| graph/iid n=1000 build+activate | — | — | — | 12599375 ns / -1 | 12506667 ns / -1 | — | 12425542 ns / -1 |
| graph/iid n=10000 per-iteration | — | — | — | 5693971 ns / -1 | 4638088 ns / -1 | — | 4663333 ns / -1 |
| graph/iid n=10000 build+activate | — | — | — | 131018875 ns / -1 | 130754667 ns / -1 | — | 128832083 ns / -1 |
| graph/iid n=1000 no-FE per-iteration | — | — | — | 333688 ns / -1 | 263538 ns / -1 | — | 259325 ns / -1 |
| graph/iid n=1000 no-FE build+activate | — | — | — | 11293916 ns / -1 | 11292667 ns / -1 | — | 11251292 ns / -1 |
| graph/chain n=300 per-iteration | — | — | — | 1067021 ns / -1 | 994004 ns / -1 | — | 993412 ns / -1 |
| graph/chain n=300 build+activate | — | — | — | 11540958 ns / -1 | 10888625 ns / -1 | — | 10755667 ns / -1 |
| graph/chain n=1000 per-iteration | — | — | — | 4077938 ns / -1 | 3812471 ns / -1 | — | 3726642 ns / -1 |
| graph/chain n=1000 build+activate | — | — | — | 39983250 ns / -1 | 39041958 ns / -1 | — | 38319167 ns / -1 |
| graph/chain n=1000 no-FE per-iteration | — | — | — | 3102633 ns / -1 | 2733838 ns / -1 | — | 2730654 ns / -1 |
| graph/chain n=1000 no-FE build+activate | — | — | — | 35627250 ns / -1 | 34196625 ns / -1 | — | 33718000 ns / -1 |
