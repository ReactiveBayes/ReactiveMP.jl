"""
    BinomialPolya

A binomial regression through the logistic, `y ~ Binomial(n, σ(xᵀβ))`, with the weights `β`
Pólya-Gamma augmented so that their message is normal. Its interfaces are `y`, the count, `x`,
the covariates, `n`, the number of trials, and `β`, the weights.

Its rules run under its own algorithm, [`BinomialPolyaApproximation`](@ref). The rule towards
`β` reads the message on `β` itself, beside the marginals the factorisation gives, so a model
initialises that message.
"""
struct BinomialPolya end

"""
    BinomialPolyaApproximation(; samples = nothing)

[`BinomialPolya`](@ref)'s algorithm. With `samples = nothing`, the default, the rules use the
means: the Pólya-Gamma mean at `xᵀ` times the mean of `β`, and `σ(xᵀβ)` at it. With a number of
samples they average over that many draws of `β` from the rule context's generator, `ctx.rng`,
which the caller owns.

The average energy does not sample: `xᵀβ` is normal under a normal `q(β)`, so its expectation is
computed by Gauss–Hermite cubature.
"""
struct BinomialPolyaApproximation{S <: Union{Nothing, Int}} <: AbstractAlgorithm
    samples::S
end

BinomialPolyaApproximation(; samples = nothing) = BinomialPolyaApproximation(samples)

@define_factor_node(node = BinomialPolya, type = Stochastic, interfaces = [:y, :x, :n, :β], algorithm = BinomialPolyaApproximation)

# Declared on the parametric type, for both of its variants: the node's own `algorithm` keyword
# binds its dependencies and its rules without `algorithm` to its default instance's type,
# `BinomialPolyaApproximation{Nothing}`.
@define_dependencies(
    node = BinomialPolya, algorithm = BinomialPolyaApproximation,
    dependencies = [:y => (default,), :x => (default,), :n => (default,), :β => (default, m[:β])],
)

# `k` draws of `β` from `d`, each a sample: the columns of a multivariate draw, the entries of a
# univariate one.
draws(rng, d::MultivariateDistribution, k) = eachcol(rand(rng, d, k))
draws(rng, d::UnivariateDistribution, k) = rand(rng, d, k)

# Towards `β`: the Pólya-Gamma augmented likelihood, normal in `β`, with precision x ω xᵀ and
# weighted mean (y - n/2) x, ω the Pólya-Gamma mean at xᵀβ.
function binomial_polya_towards_β(ω, q_y, q_x, q_n, m_β)
    y, x, n = mean(q_y), mean(q_x), mean(q_n)
    T = promote_samplefloattype(q_y, q_x, q_n, m_β)
    ξ = convert(T, y - n / 2) * x
    return convert(promote_variate_type(typeof(ξ), NormalWeightedMeanPrecision), ξ, x * convert(T, ω) * x')
end

@define_message_update_rule(
    node = BinomialPolya, target = :β, algorithm = BinomialPolyaApproximation{Nothing},
    args = (q[:y]::Union{PointMass, Multinomial}, q[:x]::PointMass, q[:n]::PointMass, m[:β]::NormalDistributionsFamily),
    body = (args) -> begin
        ω = mean(PolyaGammaHybridSampler(mean(args.q[:n]), dot(mean(args.q[:x]), mean(args.m[:β]))))
        binomial_polya_towards_β(ω, args.q[:y], args.q[:x], args.q[:n], args.m[:β])
    end,
)

@define_message_update_rule(
    node = BinomialPolya, target = :β, algorithm = BinomialPolyaApproximation{Int}, ctx = (:rng,),
    args = (q[:y]::Union{PointMass, Multinomial}, q[:x]::PointMass, q[:n]::PointMass, m[:β]::NormalDistributionsFamily),
    body = (algo, ctx, args) -> begin
        x, n = mean(args.q[:x]), mean(args.q[:n])
        βs = draws(ctx.rng, args.m[:β], algo.samples)
        ω = mean(β -> rand(ctx.rng, PolyaGammaHybridSampler(n, dot(x, β))), βs)
        binomial_polya_towards_β(ω, args.q[:y], args.q[:x], args.q[:n], args.m[:β])
    end,
)

# Towards `y`: the binomial at σ(xᵀβ), at the mean of `β` or averaged over draws of it.
@define_message_update_rule(
    node = BinomialPolya, target = :y, algorithm = BinomialPolyaApproximation{Nothing},
    args = (q[:x]::PointMass, q[:n]::Any, q[:β]::NormalDistributionsFamily),
    body = (args) -> Binomial(mean(args.q[:n]), logistic(dot(mean(args.q[:x]), mean(args.q[:β])))),
)

@define_message_update_rule(
    node = BinomialPolya, target = :y, algorithm = BinomialPolyaApproximation{Int}, ctx = (:rng,),
    args = (q[:x]::PointMass, q[:n]::Any, q[:β]::NormalDistributionsFamily),
    body = (algo, ctx, args) -> begin
        x = mean(args.q[:x])
        βs = draws(ctx.rng, args.q[:β], algo.samples)
        Binomial(mean(args.q[:n]), mean(β -> logistic(dot(x, β)), βs))
    end,
)

const BINOMIAL_POLYA_CUBATURE_POINTS = 32

# ⟨-log p(y | n, x, β)⟩ = -log C(n, y) - y ⟨ψ⟩ + n ⟨softplus(ψ)⟩ with ψ = xᵀβ, normal under a normal
# q(β). softplus at the mean of ψ would be biased low, since softplus is convex, so the
# expectation is computed by cubature.
@define_average_energy(
    node = BinomialPolya, algorithm = BinomialPolyaApproximation,
    args = (q[:y]::PointMass, q[:x]::PointMass, q[:n]::PointMass, q[:β]::Any),
    body = (args) -> begin
        y, x, n = mean(args.q[:y]), mean(args.q[:x]), mean(args.q[:n])
        mβ, Vβ = mean_cov(args.q[:β])
        mψ, vψ = dot(x, mβ), dot(x, Vβ * x)
        gh = ghcubature(BINOMIAL_POLYA_CUBATURE_POINTS)
        softplus_ψ = sum(((w, p),) -> w * softplus(p), zip(getweights(gh, mψ, vψ), getpoints(gh, mψ, vψ)))
        -(loggamma(n + 1) - loggamma(n - y + 1) - loggamma(y + 1)) - y * mψ + n * softplus_ψ
    end,
)
