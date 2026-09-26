"""
    BinomialPolya

The stochastic node of a binomial regression through the logistic,

```math
p(y \\mid x, n, \\beta) = \\binom{n}{y} \\sigma(x^\\top \\beta)^{y} \\big(1 - \\sigma(x^\\top \\beta)\\big)^{n - y},
\\qquad \\sigma(\\psi) = \\frac{1}{1 + e^{-\\psi}},
```

with its weights `β` Pólya-Gamma augmented so that their message is normal. With `n = 1` it is
logistic regression.

# Interfaces

- `y`: the number of successes, observed, a `PointMass`;
- `x`: the covariates, observed, a `PointMass` holding a vector (or a number, for a scalar `β`);
- `n`: the number of trials, observed, a `PointMass`;
- `β`: the weights, with a normal message and marginal, `MvNormal` for a vector `x`.

# Augmentation

$(DOC_POLYA_AUGMENTATION)

With `ψ = xᵀβ` and `b = n`, the message towards `β` has the weighted mean `(y - n/2) x` and the
precision `ω x xᵀ`, where `ω` is the Pólya-Gamma mean at `xᵀ` times the mean of the message on
`β`. The message towards `y` is `Binomial(n, σ(xᵀ mᵦ))`, `mᵦ` the mean of `q(β)`. Both rules
average over draws of `β` instead when [`BinomialPolyaApproximation`](@ref) is given a number of
samples.

The average energy is that of the binomial likelihood itself, not of the augmented one:
`-log C(n, y) - y ⟨ψ⟩ + n ⟨softplus(ψ)⟩`, with `⟨softplus(ψ)⟩` under the normal `ψ = xᵀβ` by
Gauss–Hermite cubature with a fixed 32 points.

# Limitations

- The rule towards `β` reads the **message on `β`** itself, besides the marginals its
  factorisation gives it, so a model must initialise that message.
- `y`, `x` and `n` must be observed: their rules and the average energy take `PointMass`
  marginals only, and there is no rule towards `x` or `n`.
- The rule towards `β` also takes a `Multinomial` `q(y)` by its type, and fails on it, since the
  mean of a `Multinomial` is a vector.

# Examples

```jldoctest; setup = :(using PolyaMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily)
julia> result = @call_message_update_rule(
           node = BinomialPolya, target = :y,
           q = (x = PointMass([1.0, 2.0]), n = PointMass(10), β = MvNormalMeanCovariance([0.0, 0.0], [1.0 0.0; 0.0 1.0])),
       );

julia> mean(getresult(result)) ≈ 5.0
true
```

At `β = 0` the success probability is `σ(0) = 1/2`, so the message is `Binomial(10, 1/2)`.

See also [`BinomialPolyaApproximation`](@ref), [`MultinomialPolya`](@ref).
"""
struct BinomialPolya end

"""
    BinomialPolyaApproximation(; samples = nothing)

The algorithm of [`BinomialPolya`](@ref), and its default: a model names it only to sample.

# Keywords

- `samples`: `nothing` or a number of draws. Default `nothing`. With `nothing`, the rules use
  means: the Pólya-Gamma mean at `xᵀ` times the mean of `β`, and `σ(xᵀβ)` at the mean of `β`.
  With a number `k`, they average over `k` draws of `β`, the rule towards `β` over one
  Pólya-Gamma draw for each, from the rule context's generator, `ctx.rng`. An engine supplies
  it; a call by hand passes `ctx = MessagePassingRulesBase.RuleContext(rng = …)`.

The average energy does not sample under either: `xᵀβ` is normal under a normal `q(β)`, so its
expectation is computed by Gauss–Hermite cubature.

# Examples

```jldoctest; setup = :(using PolyaMessagePassingRules)
julia> BinomialPolyaApproximation().samples === nothing
true

julia> BinomialPolyaApproximation(samples = 100).samples
100
```
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

# The number of Gauss–Hermite points of the average energy, fixed: the algorithm has no keyword
# for it.
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
