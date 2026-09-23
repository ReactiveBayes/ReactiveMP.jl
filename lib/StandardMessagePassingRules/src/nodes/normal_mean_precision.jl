@define_factor_node(
    node = NormalMeanPrecision,
    type = Stochastic,
    interfaces = [:out, (:μ, aliases = [:mean]), (:τ, aliases = [:invcov, :precision])],
)

# Also each component's energy inside `NormalMixture`.
function normal_mean_precision_energy(q_out, q_μ, q_τ)
    μ_mean, μ_var = mean_var(q_μ)
    out_mean, out_var = mean_var(q_out)
    return (log2π - mean(log, q_τ) + mean(q_τ) * (μ_var + out_var + abs2(μ_mean - out_mean))) / 2
end

@define_average_energy(
    node = NormalMeanPrecision,
    args = (q[:out]::Any, q[:μ]::Any, q[:τ]::Any),
    body = (args) -> normal_mean_precision_energy(args.q[:out], args.q[:μ], args.q[:τ]),
)

@define_average_energy(
    node = NormalMeanPrecision,
    args = (q[:out, :μ]::MultivariateNormalDistributionsFamily, q[:τ]::Any),
    body = (args) -> begin
        m, Σ = mean_cov(args.q[:out, :μ])
        (log2π - mean(log, args.q[:τ]) + mean(args.q[:τ]) * (Σ[1, 1] + Σ[2, 2] - Σ[1, 2] - Σ[2, 1] + abs2(m[1] - m[2]))) / 2
    end,
)
