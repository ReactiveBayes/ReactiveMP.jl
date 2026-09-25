# Forward messages are pushed-forward densities, so their log-scale is 0, as v6 annotated.

@define_message_update_rule(node = *, target = :out, args = (m[:A]::PointMass, m[:in]::PointMass), body = (args) -> PointMass(mean(args.m[:A]) * mean(args.m[:in])))

@define_message_update_rule(node = *, target = :out, args = (m[:A]::PointMass{<:Real}, m[:in]::GammaDistributionsFamily), body = (args) -> scaled(mean(args.m[:A]), args.m[:in]))

@define_message_update_rule(node = *, target = :out, args = (m[:A]::GammaDistributionsFamily, m[:in]::PointMass{<:Real}), body = (args) -> scaled(mean(args.m[:in]), args.m[:A]))

@define_message_update_rule(
    node = *, target = :out,
    args = (m[:A]::PointMass{<:AbstractMatrix}, m[:in]::NormalDistributionsFamily),
    body = (args, ann) -> (annotate!(ann, :logscale, 0); scaled(mean(args.m[:A]), args.m[:in])),
)

# A vector times a scalar commutes, so either factor may be the vector.
@define_message_update_rule(
    node = *, target = :out,
    args = (m[:A]::PointMass{<:AbstractVector}, m[:in]::UnivariateNormalDistributionsFamily),
    body = (args, ann) -> (annotate!(ann, :logscale, 0); scaled(mean(args.m[:A]), args.m[:in])),
)

@define_message_update_rule(
    node = *, target = :out,
    args = (m[:A]::UnivariateNormalDistributionsFamily, m[:in]::PointMass{<:AbstractVector}),
    body = (args, ann) -> (annotate!(ann, :logscale, 0); scaled(mean(args.m[:in]), args.m[:A])),
)

@define_message_update_rule(
    node = *, target = :out,
    args = (m[:A]::PointMass{<:Real}, m[:in]::NormalDistributionsFamily),
    body = (args, ann) -> (annotate!(ann, :logscale, 0); scaled(mean(args.m[:A]), args.m[:in])),
)

@define_message_update_rule(
    node = *, target = :out,
    args = (m[:A]::NormalDistributionsFamily, m[:in]::PointMass{<:Real}),
    body = (args, ann) -> (annotate!(ann, :logscale, 0); scaled(mean(args.m[:in]), args.m[:A])),
)

@define_message_update_rule(
    node = *, target = :out,
    args = (m[:A]::PointMass{<:UniformScaling}, m[:in]::NormalDistributionsFamily),
    body = (args, ann) -> (annotate!(ann, :logscale, 0); scaled(mean(args.m[:A]).λ, args.m[:in])),
)

# The product of two univariate Gaussians, in closed form.
@define_message_update_rule(
    node = *, target = :out,
    args = (m[:A]::UnivariateGaussianDistributionsFamily, m[:in]::UnivariateGaussianDistributionsFamily),
    body = (args) -> begin
        μ_A, v_A = mean_var(args.m[:A])
        μ_in, v_in = mean_var(args.m[:in])
        ContinuousUnivariateLogPdf(besselmod(μ_in, v_in, μ_A, v_A, zero(μ_A)))
    end,
)

@define_message_update_rule(
    node = *, target = :out, ctx = (:rng,),
    args = (m[:A]::UnivariateDistribution, m[:in]::UnivariateDistribution),
    body = (algo, ctx, args) -> sampled_product_logpdf(ctx.rng, args.m[:A], args.m[:in], algo.samples),
)
