using Random: rand

raw"""
    test_generic_vmp_rule(rng, log_node_function, index_to_remove, N_samples, q_s...)

Return a function of interface `i` that evaluates the log factor for each of
`N_samples` independent draws from the other interfaces. Averaging its returned
vector estimates the unnormalized full mean-field VMP log message:

```math
\widehat{\log \mu}_{f \rightarrow i}(x_i) =
\frac{1}{N}\sum_{s=1}^{N}
\log f(x_1^{(s)}, \ldots, x_i, \ldots, x_K^{(s)}),
\qquad x_j^{(s)} \sim q_j \quad (j \ne i).
```

Supply one entry in `q_s` per argument of `log_node_function`, with `nothing` at
`index_to_remove` and sampleable distributions elsewhere. `log_node_function`
must return the logarithm of the factor value.

Samples are drawn once using `rng` and reused in the same order for every
evaluation. For example, `log_samples = test_generic_vmp_rule(rng, log_f, 2, N,
q₁, nothing, q₃)` returns a function of the second argument of `log_f`.

To remove the unknown additive log constant, use paired differences
`d = log_samples(x) .- log_samples(x_reference)`. Their mean estimates the log
message difference, and `std(d) / sqrt(length(d))` estimates its Monte Carlo
standard error. Normal-approximation confidence intervals require finite
variance and sufficiently many samples.
"""
function test_generic_vmp_rule(rng, log_node_function::F, index_to_remove, N_samples, q_s...) where {F}
    samples = map(q -> isnothing(q) ? nothing : [rand(rng, q) for _ in 1:N_samples], q_s)

    return function (x)
        return map(1:N_samples) do s
            args = ntuple(length(q_s)) do j
                j == index_to_remove ? x : samples[j][s]
            end
            log_node_function(args...)
        end
    end
end
