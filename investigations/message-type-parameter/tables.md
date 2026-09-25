
## micro_results.tsv
| case | A min (median of runs; range) | B min | B/A | A allocs/bytes | B allocs/bytes |
|---|---|---|---|---|---|
| map/direct  NMV->out m[μ]::NMV m[v]::PM (BP) | 780 ns (770 ns–787 ns) | 1.55 µs (1.53 µs–1.57 µs) | 1.983 | 16 / 960 | 26 / 1440 |
| map/barrier NMV->out m[μ]::NMV m[v]::PM (BP) | 801 ns (784 ns–806 ns) | 1.54 µs (1.54 µs–1.56 µs) | 1.924 | 17 / 992 | 27 / 1472 |
| resolve/barrier NMV->out m[μ]::NMV m[v]::PM (BP) | 27 ns (26 ns–27 ns) | 665 ns (663 ns–679 ns) | 25.011 | 2 / 144 | 6 / 272 |
| deferred/barrier NMV->out m[μ]::NMV m[v]::PM (BP) | 817 ns (805 ns–821 ns) | 1.56 µs (1.53 µs–1.58 µs) | 1.908 | 17 / 992 | 27 / 1472 |
| map/direct  NMP->μ q[out]::PM q[τ]::Gamma (VMP) | 889 ns (879 ns–899 ns) | 1.84 µs (1.82 µs–1.87 µs) | 2.067 | 14 / 640 | 21 / 864 |
| map/barrier NMP->μ q[out]::PM q[τ]::Gamma (VMP) | 919 ns (915 ns–923 ns) | 1.85 µs (1.83 µs–1.89 µs) | 2.012 | 15 / 672 | 22 / 896 |
| resolve/barrier NMP->μ q[out]::PM q[τ]::Gamma (VMP) | 27 ns (26 ns–27 ns) | 1.02 µs (1.02 µs–1.04 µs) | 38.246 | 2 / 144 | 6 / 272 |
| deferred/barrier NMP->μ q[out]::PM q[τ]::Gamma (VMP) | 909 ns (902 ns–934 ns) | 1.89 µs (1.86 µs–1.89 µs) | 2.077 | 15 / 672 | 22 / 896 |
| map/direct  MvNMC->out m[μ]::MvNMC(3) m[Σ]::PM (BP) | 801 ns (783 ns–811 ns) | 1.68 µs (1.67 µs–1.70 µs) | 2.097 | 18 / 1104 | 28 / 1584 |
| map/barrier MvNMC->out m[μ]::MvNMC(3) m[Σ]::PM (BP) | 825 ns (807 ns–836 ns) | 1.68 µs (1.65 µs–1.69 µs) | 2.035 | 19 / 1136 | 29 / 1616 |
| resolve/barrier MvNMC->out m[μ]::MvNMC(3) m[Σ]::PM (BP) | 26 ns (26 ns–26 ns) | 722 ns (720 ns–738 ns) | 27.441 | 2 / 144 | 6 / 272 |
| deferred/barrier MvNMC->out m[μ]::MvNMC(3) m[Σ]::PM (BP) | 837 ns (819 ns–850 ns) | 1.70 µs (1.67 µs–1.71 µs) | 2.031 | 19 / 1136 | 29 / 1616 |
| map/direct  Bernoulli->out m[p]::Beta | 938 ns (933 ns–944 ns) | 1.37 µs (1.35 µs–1.43 µs) | 1.458 | 18 / 1024 | 23 / 1312 |
| map/barrier Bernoulli->out m[p]::Beta | 963 ns (944 ns–965 ns) | 1.39 µs (1.35 µs–1.39 µs) | 1.441 | 19 / 1056 | 24 / 1344 |
| resolve/barrier Bernoulli->out m[p]::Beta | 26 ns (26 ns–27 ns) | 715 ns (702 ns–721 ns) | 27.183 | 2 / 144 | 6 / 272 |
| deferred/barrier Bernoulli->out m[p]::Beta | 949 ns (948 ns–978 ns) | 1.38 µs (1.38 µs–1.40 µs) | 1.454 | 19 / 1056 | 24 / 1344 |
| map/direct  Categorical->p q[out]::Categorical | 1.04 µs (1.03 µs–1.07 µs) | 1.56 µs (1.55 µs–1.62 µs) | 1.500 | 21 / 1168 | 26 / 1440 |
| map/barrier Categorical->p q[out]::Categorical | 1.07 µs (1.07 µs–1.08 µs) | 1.56 µs (1.55 µs–1.57 µs) | 1.455 | 22 / 1200 | 27 / 1472 |
| resolve/barrier Categorical->p q[out]::Categorical | 27 ns (27 ns–27 ns) | 820 ns (816 ns–833 ns) | 30.582 | 2 / 144 | 7 / 400 |
| deferred/barrier Categorical->p q[out]::Categorical | 1.08 µs (1.07 µs–1.10 µs) | 1.60 µs (1.59 µs–1.61 µs) | 1.473 | 22 / 1200 | 27 / 1472 |
| joint-marginal/barrier NMV (out,μ) | 150 ns (148 ns–155 ns) | 1.27 µs (1.26 µs–1.27 µs) | 8.445 | 11 / 544 | 19 / 880 |
| product/NMV x3 | 1.32 µs (1.30 µs–1.33 µs) | 1.33 µs (1.32 µs–1.35 µs) | 1.009 | 12 / 384 | 13 / 400 |
| product/deferred NMV x3 | 1.36 µs (1.34 µs–1.38 µs) | 1.35 µs (1.33 µs–1.35 µs) | 0.988 | 12 / 384 | 13 / 400 |
| product/NMV x10 | 1.48 µs (1.46 µs–1.49 µs) | 4.02 µs (4.02 µs–4.17 µs) | 2.721 | 33 / 944 | 48 / 1408 |
| product/deferred NMV x10 | 1.69 µs (1.67 µs–1.70 µs) | 4.10 µs (4.03 µs–4.11 µs) | 2.430 | 33 / 944 | 48 / 1408 |
| product/MvNMC(3) x3 | 2.15 µs (2.09 µs–2.15 µs) | 2.11 µs (2.11 µs–2.12 µs) | 0.983 | 34 / 1648 | 35 / 1664 |
| marginal-at-variable/NMV x3 | 1.55 µs (1.54 µs–1.60 µs) | 1.38 µs (1.34 µs–1.43 µs) | 0.889 | 14 / 464 | 14 / 432 |
| stream/marginal combineLatest x3 round | 108 ns (106 ns–110 ns) | 114 ns (108 ns–116 ns) | 1.055 | 0 / 0 | 0 / 0 |
| stream/message combineLatest x3 round | 110 ns (109 ns–110 ns) | 116 ns (108 ns–116 ns) | 1.049 | 0 / 0 | 0 / 0 |
| graph/iid n=1000 10 it | 12.11 ms (12.00 ms–12.22 ms) | 44.08 ms (42.33 ms–44.20 ms) | 3.640 | -1 / 13446112 | -1 / 23212224 |
| graph/chain n=300 10 it | 23.99 ms (23.80 ms–24.32 ms) | 68.02 ms (66.52 ms–69.14 ms) | 2.835 | -1 / 20297952 | -1 / 31938304 |

## model_results.tsv
| case | A min (median of runs; range) | B min | B/A | A allocs/bytes | B allocs/bytes |
|---|---|---|---|---|---|
| ssm1 load | 4662.0 ms (4588.6 ms–4920.8 ms) | 4614.7 ms (4157.5 ms–4777.9 ms) | 0.990 |  / 0 |  / 0 |
| ssm1 ttfx | 11.15 s (11.10 s–11.25 s) | 11.27 s (11.14 s–11.43 s) | 1.010 |  / 0 |  / 0 |
| ssm1 infer(I=1) | 111.9 ms (111.6 ms–112.6 ms) | 126.9 ms (125.1 ms–128.2 ms) | 1.134 |  / 81456624 |  / 86033152 |
| ssm2 load | 4920.0 ms (4633.7 ms–4958.7 ms) | 4727.1 ms (4618.7 ms–4869.1 ms) | 0.961 |  / 0 |  / 0 |
| ssm2 ttfx | 14.74 s (14.71 s–15.03 s) | 14.72 s (14.58 s–14.79 s) | 0.998 |  / 0 |  / 0 |
| ssm2 infer(I=1) | 190.7 ms (189.5 ms–191.2 ms) | 217.2 ms (214.8 ms–219.6 ms) | 1.139 |  / 136429440 |  / 143694912 |
| iid load | 4886.4 ms (4581.3 ms–4886.8 ms) | 4643.1 ms (4607.9 ms–4805.8 ms) | 0.950 |  / 0 |  / 0 |
| iid ttfx | 9138.6 ms (9078.5 ms–9273.6 ms) | 9235.3 ms (9127.3 ms–9396.8 ms) | 1.011 |  / 0 |  / 0 |
| iid infer(I=10) | 65.5 ms (65.4 ms–65.7 ms) | 100.6 ms (99.0 ms–100.7 ms) | 1.536 |  / 59827408 |  / 73626304 |
| iid infer(I=20) | 92.6 ms (90.9 ms–94.2 ms) | 163.0 ms (161.8 ms–164.4 ms) | 1.760 |  / 87219600 |  / 114945536 |
| iid per-iteration | 2.7 ms (2.6 ms–2.9 ms) | 6.3 ms (6.2 ms–6.4 ms) | 2.336 |  / 2739219 |  / 4131923 |
| nl load | 4610.4 ms (4596.5 ms–4709.5 ms) | 4660.4 ms (4657.9 ms–4851.6 ms) | 1.011 |  / 0 |  / 0 |
| nl ttfx | 14.83 s (14.82 s–14.86 s) | 14.82 s (14.73 s–14.99 s) | 0.999 |  / 0 |  / 0 |
| nl infer(I=5) | 82.1 ms (81.6 ms–82.5 ms) | 137.4 ms (137.0 ms–137.7 ms) | 1.673 |  / 56792416 |  / 67903392 |
| nl infer(I=10) | 111.1 ms (109.2 ms–113.1 ms) | 218.2 ms (215.4 ms–219.7 ms) | 1.963 |  / 75975888 |  / 97805824 |
| nl per-iteration | 5.7 ms (5.4 ms–6.3 ms) | 16.2 ms (15.5 ms–16.5 ms) | 2.816 |  / 3836694 |  / 5980486 |
| hmm load | 4761.5 ms (4716.8 ms–4914.5 ms) | 4862.1 ms (4676.7 ms–4883.1 ms) | 1.021 |  / 0 |  / 0 |
| hmm ttfx | 15.50 s (15.49 s–15.63 s) | 15.49 s (15.49 s–15.52 s) | 0.999 |  / 0 |  / 0 |
| hmm infer(I=10) | 86.5 ms (86.5 ms–86.8 ms) | 138.2 ms (136.3 ms–139.2 ms) | 1.597 |  / 75092752 |  / 95807504 |
| hmm infer(I=20) | 140.1 ms (136.4 ms–140.4 ms) | 239.1 ms (238.2 ms–239.4 ms) | 1.706 |  / 122485280 |  / 164124464 |
| hmm per-iteration | 5.4 ms (5.0 ms–5.4 ms) | 10.1 ms (9.9 ms–10.3 ms) | 1.888 |  / 4739252 |  / 6831696 |
