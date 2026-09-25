# Towards A from a known `in`, which must commute with A: a scalar, or a vector for a scalar A.
# A matrix `in` is refused: in * A is not A * in.

@define_message_update_rule(
    node = *, target = :A,
    args = (m[:out]::PointMass, m[:in]::PointMass{<:Union{Real, AbstractVector}}),
    body = (args) -> unscaled(nothing, args.m[:out], mean(args.m[:in])),
)

# m(a) = m_out(a c) for a known scalar c integrates to |c|^(-d), as towards `in`.
@define_message_update_rule(node = *, target = :A, args = (m[:out]::GammaDistributionsFamily, m[:in]::PointMass{<:Real}), logscale = (args) -> unscaled_logscale(args.m[:out], mean(args.m[:in])), body = (args) -> unscaled(nothing, args.m[:out], mean(args.m[:in])))

@define_message_update_rule(
    node = *, target = :A, ctx = (:matrix_correction,),
    args = (m[:out]::MultivariateNormalDistributionsFamily, m[:in]::PointMass{<:AbstractVector}),
    body = (ctx, args) -> unscaled(ctx, args.m[:out], mean(args.m[:in])),
)

@define_message_update_rule(
    node = *, target = :A, ctx = (:matrix_correction,),
    args = (m[:out]::NormalDistributionsFamily, m[:in]::PointMass{<:Real}),
    logscale = (args) -> unscaled_logscale(args.m[:out], mean(args.m[:in])),
    body = (ctx, args) -> unscaled(ctx, args.m[:out], mean(args.m[:in])),
)

@define_message_update_rule(
    node = *, target = :A,
    args = (m[:out]::UnivariateGaussianDistributionsFamily, m[:in]::UnivariateGaussianDistributionsFamily),
    body = (args) -> gaussian_ratio_logpdf(args.m[:out], args.m[:in]),
)

@define_message_update_rule(
    node = *, target = :A, ctx = (:rng,),
    args = (m[:out]::UnivariateDistribution, m[:in]::UnivariateDistribution),
    body = (algo, ctx, args) -> sampled_ratio_logpdf(ctx.rng, args.m[:out], args.m[:in], algo.samples),
)
