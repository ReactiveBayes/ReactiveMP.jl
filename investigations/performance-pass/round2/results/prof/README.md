`prof_setup.jl` outputs: inclusive milliseconds per function of one `infer` at round 1's iteration
count (or 200 iterations, `*_iid200`), compared with `../../cmp_prof.py`. `v6_*` is RxInfer 5.5.2 over
ReactiveMP 6.5.0, `FULLT_*` round 1's recommended set, `Q_*` the prototypes Q1 and Q3 (before the
cached dependencies), `Q2_*` Q1, Q3, Q3b and Q4 (before Q5, the mutable mapping). Indicative: the
machine was in use.
`R_*`: R, the round-2 set before P4 was reduced and before the plan was keyed on the declaration.
