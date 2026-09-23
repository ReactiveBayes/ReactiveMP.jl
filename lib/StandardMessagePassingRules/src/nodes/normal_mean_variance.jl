@define_factor_node(
    node = NormalMeanVariance,
    type = Stochastic,
    interfaces = [:out, (:μ, aliases = [:mean]), (:v, aliases = [:var])],
    algorithm = BP,
)

@define_average_energy(
    node = NormalMeanVariance,
    args = (q[:out]::Any, q[:μ]::Any, q[:v]::Any),
    body = (args) -> begin
        μ_mean, μ_var = mean_var(args.q[:μ])
        out_mean, out_var = mean_var(args.q[:out])
        (log2π + mean(log, args.q[:v]) + mean(inv, args.q[:v]) * (μ_var + out_var + abs2(μ_mean - out_mean))) / 2
    end,
)

@define_average_energy(
    node = NormalMeanVariance,
    args = (q[:out, :μ]::MultivariateNormalDistributionsFamily, q[:v]::Any),
    body = (args) -> begin
        m, Σ = mean_cov(args.q[:out, :μ])
        (log2π + mean(log, args.q[:v]) + mean(inv, args.q[:v]) * (Σ[1, 1] + Σ[2, 2] - Σ[1, 2] - Σ[2, 1] + abs2(m[1] - m[2]))) / 2
    end,
)
