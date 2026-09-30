@define_message_update_rule(
    node = NormalMeanVariance, target = :μ,
    args = (m[:out]::PointMass, m[:v]::PointMass),
    logscale = 0,
    body = (args) -> NormalMeanVariance(mean(args.m[:out]), mean(args.m[:v])),
)

@define_message_update_rule(
    node = NormalMeanVariance, target = :μ,
    args = (m[:out]::UnivariateNormalDistributionsFamily, m[:v]::PointMass),
    logscale = 0,
    body = (args) -> begin
        out_mean, out_var = mean_var(args.m[:out])
        NormalMeanVariance(out_mean, out_var + mean(args.m[:v]))
    end,
)

@define_message_update_rule(
    node = NormalMeanVariance, target = :μ,
    args = (q[:out]::PointMass, q[:v]::PointMass),
    logscale = 0,
    body = (args) -> NormalMeanVariance(mean(args.q[:out]), mean(args.q[:v])),
)

@define_message_update_rule(
    node = NormalMeanVariance, target = :μ,
    args = (q[:out]::Any, q[:v]::Any),
    body = (args) -> NormalMeanVariance(mean(args.q[:out]), gaussian_variational_variance(args.q[:v])),
)

@define_message_update_rule(
    node = NormalMeanVariance, target = :μ,
    args = (m[:out]::PointMass, q[:v]::Any),
    body = (args) -> NormalMeanVariance(mean(args.m[:out]), gaussian_variational_variance(args.q[:v])),
)

# No log scale: `q_v` may be any marginal. Towards `:out`, a rule of its own for a `PointMass`
# `q_v` declares 0; this target has no such rule.
@define_message_update_rule(
    node = NormalMeanVariance, target = :μ,
    args = (m[:out]::UnivariateNormalDistributionsFamily, q[:v]::Any),
    body = (args) -> begin
        out_mean, out_var = mean_var(args.m[:out])
        NormalMeanVariance(out_mean, out_var + gaussian_variational_variance(args.q[:v]))
    end,
)
