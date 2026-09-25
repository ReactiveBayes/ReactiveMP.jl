@define_message_update_rule(
    node = NormalMeanPrecision, target = :μ,
    args = (m[:out]::PointMass, m[:τ]::PointMass),
    logscale = 0,
    body = (args) -> NormalMeanPrecision(mean(args.m[:out]), mean(args.m[:τ])),
)

@define_message_update_rule(
    node = NormalMeanPrecision, target = :μ,
    args = (m[:out]::UnivariateNormalDistributionsFamily, m[:τ]::PointMass),
    logscale = 0,
    body = (args) -> begin
        out_mean, out_var = mean_var(args.m[:out])
        NormalMeanVariance(out_mean, out_var + inv(mean(args.m[:τ])))
    end,
)

@define_message_update_rule(
    node = NormalMeanPrecision, target = :μ,
    args = (q[:out]::PointMass, q[:τ]::PointMass),
    logscale = 0,
    body = (args) -> NormalMeanPrecision(mean(args.q[:out]), mean(args.q[:τ])),
)

@define_message_update_rule(
    node = NormalMeanPrecision, target = :μ,
    args = (q[:out]::Any, q[:τ]::Any),
    body = (args) -> NormalMeanPrecision(mean(args.q[:out]), mean(args.q[:τ])),
)

@define_message_update_rule(
    node = NormalMeanPrecision, target = :μ,
    args = (m[:out]::PointMass, q[:τ]::Any),
    body = (args) -> NormalMeanPrecision(mean(args.m[:out]), mean(args.q[:τ])),
)

@define_message_update_rule(
    node = NormalMeanPrecision, target = :μ,
    args = (m[:out]::UnivariateNormalDistributionsFamily, q[:τ]::Any),
    body = (args) -> begin
        out_mean, out_var = mean_var(args.m[:out])
        NormalMeanVariance(out_mean, out_var + inv(mean(args.q[:τ])))
    end,
)
