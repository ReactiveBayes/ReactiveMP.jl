@define_message_update_rule(node = *, target = :in, args = (m[:out]::PointMass, m[:A]::PointMass), body = (args) -> unscaled(nothing, args.m[:out], mean(args.m[:A])))

@define_message_update_rule(node = *, target = :in, args = (m[:out]::GammaDistributionsFamily, m[:A]::PointMass{<:Real}), body = (args) -> unscaled(nothing, args.m[:out], mean(args.m[:A])))

@define_message_update_rule(
    node = *, target = :in, ctx = (:matrix_correction,),
    args = (m[:out]::MultivariateNormalDistributionsFamily, m[:A]::PointMass{<:Union{AbstractMatrix, AbstractVector}}),
    body = (ctx, args) -> unscaled(ctx, args.m[:out], mean(args.m[:A])),
)

@define_message_update_rule(
    node = *, target = :in, ctx = (:matrix_correction,),
    args = (m[:out]::NormalDistributionsFamily, m[:A]::PointMass{<:Real}),
    body = (ctx, args, ann) -> begin
        a = mean(args.m[:A])
        annotate!(ann, :logscale, unscaled_logscale(args.m[:out], a))
        unscaled(ctx, args.m[:out], a)
    end,
)

@define_message_update_rule(
    node = *, target = :in, ctx = (:matrix_correction,),
    args = (m[:out]::NormalDistributionsFamily, m[:A]::PointMass{<:UniformScaling}),
    body = (ctx, args, ann) -> begin
        λ = mean(args.m[:A]).λ
        annotate!(ann, :logscale, unscaled_logscale(args.m[:out], λ))
        unscaled(ctx, args.m[:out], λ)
    end,
)

@define_message_update_rule(
    node = *, target = :in,
    args = (m[:out]::UnivariateGaussianDistributionsFamily, m[:A]::UnivariateGaussianDistributionsFamily),
    body = (args) -> gaussian_ratio_logpdf(args.m[:out], args.m[:A]),
)

@define_message_update_rule(
    node = *, target = :in, ctx = (:rng,),
    args = (m[:out]::UnivariateDistribution, m[:A]::UnivariateDistribution),
    body = (algo, ctx, args) -> sampled_ratio_logpdf(ctx.rng, args.m[:out], args.m[:A], algo.samples),
)
