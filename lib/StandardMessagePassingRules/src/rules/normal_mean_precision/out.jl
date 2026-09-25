@define_message_update_rule(
    node = NormalMeanPrecision, target = :out,
    args = (m[:μ]::PointMass, m[:τ]::PointMass),
    logscale = 0,
    body = (args) -> NormalMeanPrecision(mean(args.m[:μ]), mean(args.m[:τ])),
)

@define_message_update_rule(
    node = NormalMeanPrecision, target = :out,
    args = (m[:μ]::UnivariateNormalDistributionsFamily, m[:τ]::PointMass),
    logscale = 0,
    body = (args) -> begin
        μ_mean, μ_var = mean_var(args.m[:μ])
        NormalMeanPrecision(μ_mean, inv(μ_var + inv(mean(args.m[:τ]))))
    end,
)

@define_message_update_rule(
    node = NormalMeanPrecision, target = :out,
    args = (q[:μ]::PointMass, q[:τ]::PointMass),
    logscale = 0,
    body = (args) -> NormalMeanPrecision(mean(args.q[:μ]), mean(args.q[:τ])),
)

@define_message_update_rule(
    node = NormalMeanPrecision, target = :out,
    args = (q[:μ]::Any, q[:τ]::Any),
    body = (args) -> NormalMeanPrecision(mean(args.q[:μ]), mean(args.q[:τ])),
)

@define_message_update_rule(
    node = NormalMeanPrecision, target = :out,
    args = (m[:μ]::PointMass, q[:τ]::Any),
    body = (args) -> NormalMeanPrecision(mean(args.m[:μ]), mean(args.q[:τ])),
)

@define_message_update_rule(
    node = NormalMeanPrecision, target = :out,
    args = (m[:μ]::UnivariateNormalDistributionsFamily, q[:τ]::Any),
    body = (args) -> begin
        μ_mean, μ_var = mean_var(args.m[:μ])
        NormalMeanPrecision(μ_mean, inv(μ_var + inv(mean(args.q[:τ]))))
    end,
)
