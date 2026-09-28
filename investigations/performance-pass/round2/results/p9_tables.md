| model | variant | I | T(I) | T(I) no GC | T(2I) no GC | per iteration (no GC) | setup (no GC) | bytes T(I) | first inference | T(I) / RnoP9 per round: min, median, max |
|---|---|---|---|---|---|---|---|---|---|---|
| ssm1 | RnoP9 | 1 | 40.81 ms | 40.81 ms | — | — | — | 53.6 MB | 10.16 s |  |
| ssm1 | RP9args | 1 | 44.04 ms | 44.04 ms | — | — | — | 53.6 MB | 10.52 s | 1.042, 1.063, 1.103 **slower** |
| ssm1 | R | 1 | 44.21 ms | 44.21 ms | — | — | — | 54.5 MB | 10.51 s | 1.066, 1.077, 1.083 **slower** |
| betabern | RnoP9 | 1 | 41.83 ms | 41.83 ms | — | — | — | 67.8 MB | 5.66 s |  |
| betabern | RP9args | 1 | 42.44 ms | 42.44 ms | — | — | — | 67.8 MB | 5.79 s | 0.991, 1.013, 1.016 |
| betabern | R | 1 | 42.78 ms | 42.78 ms | — | — | — | 68.5 MB | 5.83 s | 0.973, 1.023, 1.033 |
| gmm | RnoP9 | 20 | 50.09 ms | 50.09 ms | 83.27 ms | 1.66 ms | 16.91 ms | 61.8 MB | 9.68 s |  |
| gmm | RP9args | 20 | 50.10 ms | 50.10 ms | 83.68 ms | 1.68 ms | 16.52 ms | 61.8 MB | 9.98 s | 0.996, 1.000, 1.002 |
| gmm | R | 20 | 50.39 ms | 50.39 ms | 83.21 ms | 1.64 ms | 17.56 ms | 62.3 MB | 9.98 s | 1.006, 1.017, 1.024 **slower** |
