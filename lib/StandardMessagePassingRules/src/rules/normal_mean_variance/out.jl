# The variance a variational `q_v` contributes is `1/E[1/v]`, which naive VMP,
# `exp E_q[log N(out | μ, v)]`, gives. For a point mass it is `E[v]`.
variational_variance(q_v) = inv(mean(inv, q_v))

@define_message_update_rule(
    node = NormalMeanVariance, target = :out,
    args = (m[:μ]::PointMass, m[:v]::PointMass),
    logscale = 0,
    body = (args) -> NormalMeanVariance(mean(args.m[:μ]), mean(args.m[:v])),
)

@define_message_update_rule(
    node = NormalMeanVariance, target = :out,
    args = (m[:μ]::UnivariateNormalDistributionsFamily, m[:v]::PointMass),
    logscale = 0,
    body = (args) -> begin
        μ_mean, μ_var = mean_var(args.m[:μ])
        NormalMeanVariance(μ_mean, μ_var + mean(args.m[:v]))
    end,
)

@define_message_update_rule(
    node = NormalMeanVariance, target = :out,
    args = (q[:μ]::PointMass, q[:v]::PointMass),
    logscale = 0,
    body = (args) -> NormalMeanVariance(mean(args.q[:μ]), mean(args.q[:v])),
)

@define_message_update_rule(
    node = NormalMeanVariance, target = :out,
    args = (q[:μ]::Any, q[:v]::Any),
    body = (args) -> NormalMeanVariance(mean(args.q[:μ]), variational_variance(args.q[:v])),
)

@define_message_update_rule(
    node = NormalMeanVariance, target = :out,
    args = (m[:μ]::PointMass, q[:v]::Any),
    body = (args) -> NormalMeanVariance(mean(args.m[:μ]), variational_variance(args.q[:v])),
)

@define_message_update_rule(
    node = NormalMeanVariance, target = :out,
    args = (m[:μ]::UnivariateNormalDistributionsFamily, q[:v]::Any),
    body = (args) -> begin
        μ_mean, μ_var = mean_var(args.m[:μ])
        NormalMeanVariance(μ_mean, μ_var + variational_variance(args.q[:v]))
    end,
)

@define_message_update_rule(
    node = NormalMeanVariance, target = :out,
    args = (m[:μ]::UnivariateNormalDistributionsFamily, q[:v]::PointMass),
    logscale = 0,
    body = (args) -> begin
        μ_mean, μ_var = mean_var(args.m[:μ])
        NormalMeanVariance(μ_mean, μ_var + mean(args.q[:v]))
    end,
)
