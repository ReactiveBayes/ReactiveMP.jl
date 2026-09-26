"""
    ProbitMessagePassingRules

The [`Probit`](@ref) node and its rules. The node links a binary output to a real latent input
through the standard normal CDF `Φ`,

    p(out | in) = Φ(in)^out ⋅ (1 - Φ(in))^(1 - out),

the probit link of a binary classifier. `out` is observed as `0` or `1`, or given as the
probability of a `1`; `in` is a univariate normal.

The node has rules under two algorithms:

- [`ProbitEP`](@ref), the node's own and the one it runs when a model names none: expectation
  propagation, a normal message towards `in` from the tilted distribution of the message on
  `in` itself;
- [`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm), when a model names it:
  plain rules, whose message towards `in` is the exact likelihood as a log-density rather than a
  normal.

Both have an average energy, computed by
[`GaussHermiteCubature`](@extref MessagePassingRulesApproximations.GaussHermiteCubature). No rule
declares a log scale.

# Examples

```jldoctest
julia> using MessagePassingRulesBase, ExponentialFamily, BayesBase

julia> result = @call_message_update_rule(node = Probit, target = :out, m = (in = NormalMeanVariance(1.0, 0.5),));

julia> mean(getresult(result)) ≈ 0.7928919108787374   # Φ(1 / √(1 + 0.5))
true

julia> result = @call_message_update_rule(
           node = Probit, target = :in, m = (out = PointMass(1.0), in = NormalMeanPrecision(0.0, 1.0)),
       );

julia> getresult(result) isa NormalWeightedMeanPrecision
true
```
"""
module ProbitMessagePassingRules

using MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions
using MessagePassingRulesBase: AbstractAlgorithm, DefaultAlgorithm
using MessagePassingRulesApproximations: GaussHermiteCubature
using BayesBase: tiny, huge
using StatsFuns: normcdf, normlogcdf, normlogccdf, normlogpdf, logsumexp

export Probit, ProbitEP

"""
    Probit

The probit node, `out ~ Bernoulli(Φ(in))` with `Φ` the standard normal CDF: a binary output
through a real latent input. It is stochastic, with the interfaces

- `out`: the output, observed as a `PointMass` at `0` or `1`, or
  a `Bernoulli`; its value, the probability of a `1`, lies in `[0, 1]`;
- `in`: the latent input, a univariate normal.

Its algorithm is [`ProbitEP`](@ref), and it also has rules under
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm).

Under `ProbitEP` the rule towards `in` reads the message on `in` itself, so in a graph with a
loop through `in` it needs one to start from: the node declares
`NormalMeanPrecision(0.0, 100.0)`, set at activation wherever the model sets no initial message.

# Throws

- `ArgumentError`, from the expectation-propagation rules towards `in` and the joint marginal,
  when the value on `out` lies outside `[0, 1]`.

See also [`ProbitEP`](@ref).
"""
struct Probit end

"""
    ProbitEP(p::Int)
    ProbitEP(; p = 32)

[`Probit`](@ref)'s own algorithm, expectation propagation. A model that names no algorithm for
the node runs this one.

- towards `out`: `Bernoulli(Φ(μ / √(1 + v)))` from a normal message `N(μ, v)` on `in`, or
  `Bernoulli(Φ(x))` from a point mass at `x`;
- towards `in`: the normal whose product with the message on `in` (the cavity) matches the
  mean and variance of the tilted distribution `p(out | in) ⋅ m(in)`, from the message on
  `out` and the message on `in`; its precision is clamped to be positive, and the tilted
  variance to be at most the cavity's;
- the joint marginal `q(out, in)`, for a point-mass `out` only: `out` itself and the tilted
  distribution of `in`, as a `NormalMeanVariance`;
- the average energy `E_q[-log p(out | in)]`, from `q(out)` and a normal `q(in)`.

# Keywords

- `p`: the number of Gauss–Hermite points of the average energy, a positive integer. Default
  `32`. The messages are computed in closed form and do not use it.

# Examples

```jldoctest
julia> using MessagePassingRulesBase, ExponentialFamily, BayesBase

julia> energy = @call_average_energy(
           node = Probit, algorithm = ProbitEP(p = 64), q = (out = PointMass(1.0), in = NormalMeanVariance(0.0, 1.0)),
       );

julia> getresult(energy) ≈ 1.0   # E[-log Φ(z)] for a standard normal z
true
```

See also [`Probit`](@ref), and
[`GaussHermiteCubature`](@extref MessagePassingRulesApproximations.GaussHermiteCubature) for the
cubature.
"""
struct ProbitEP <: AbstractAlgorithm
    p::Int
end

ProbitEP(; p = 32) = ProbitEP(p)

@define_factor_node(
    node = Probit, type = Stochastic, interfaces = [:out, :in], algorithm = ProbitEP,
    initial_messages = [:in => NormalMeanPrecision(0.0, 100.0)],
    dependencies = [:out => (m[:in],), :in => (m[:out], m[:in])],
)

# Towards `out`: Φ of the input, integrated against a normal message, Φ(μ / √(1 + v)).
@define_message_update_rule(
    node = Probit, target = :out, args = (m[:in]::PointMass,),
    body = (args) -> Bernoulli(normcdf(mean(args.m[:in]))),
)

@define_message_update_rule(
    node = Probit, target = :out, args = (m[:in]::UnivariateNormalDistributionsFamily,),
    body = (args) -> Bernoulli(normcdf(mean(args.m[:in]) / sqrt(1 + var(args.m[:in])))),
)

# The moments of the tilted distribution Φ-likelihood × the cavity m_in, for an output
# probability p, and the variance clamped to the cavity's.
function tilted_moments(p, m_in)
    mz, vz = mean_cov(m_in)
    zero(p) <= p <= one(p) || throw(ArgumentError("the Probit node takes a message on its output with a value between 0 and 1, got $p"))
    γ = mz / sqrt(1 + vz)
    log_mom0 = if γ > 0 && p > 0.5
        logsumexp((log(1 - p), log(2 * p - 1) + normlogccdf(-γ)))
    elseif γ <= 0 && p > 0.5
        logsumexp((log(1 - p), log(2 * p - 1) + normlogcdf(γ)))
    elseif γ > 0 && p <= 0.5
        logsumexp((log(1 - p) + normlogcdf(-γ), log(p) + normlogcdf(γ)))
    else
        logsumexp((log(1 - p) + normlogccdf(γ), log(p) + normlogcdf(γ)))
    end
    tmp = log(vz) + normlogpdf(γ) - log(1 + vz) / 2 - log_mom0
    mom1 = mz + (2 * p - 1) * exp(tmp)
    mom2 = vz + mz^2 + (2 * p - 1) * 2 * mz * exp(tmp) - (2p - 1) * γ * exp(log(vz) - log(1 + vz) / 2 + tmp)
    return mom1, clamp(mom2 - mom1^2, tiny, vz), mz, vz
end

# Towards `in`, expectation propagation: the tilted distribution's moments over the cavity.
@define_message_update_rule(
    node = Probit, target = :in, args = (m[:out]::Union{PointMass, Bernoulli}, m[:in]::UnivariateNormalDistributionsFamily),
    body = (args) -> begin
        mpz, vpz, mz, vz = tilted_moments(mean(args.m[:out]), args.m[:in])
        NormalWeightedMeanPrecision(mpz / vpz - mz / vz, clamp(1 / vpz - 1 / vz, tiny, huge))
    end,
)

# The joint of a point-mass output and `in`: the output, and the tilted distribution of `in`.
@define_marginal_update_rule(
    node = Probit, target = (:out, :in), args = (m[:out]::PointMass, m[:in]::UnivariateNormalDistributionsFamily),
    body = (args) -> begin
        mpz, vpz, _, _ = tilted_moments(mean(args.m[:out]), args.m[:in])
        cluster = FactorizedCluster((:out,) => args.m[:out], (:in,) => NormalMeanVariance(mpz, vpz))
        BayesBase.convert_paramfloattype(BayesBase.promote_paramfloattype(args.m[:out], args.m[:in]), cluster)
    end,
)

# E_q[-log p(out | in)], by Gauss–Hermite cubature over q(in) with `p` points. log Φ is computed
# directly, as `normlogcdf`: log(Φ(x)) underflows at a far point and would turn 0 ⋅ log Φ into NaN
# for a point-mass output.
function probit_energy(points, q_out, q_in)
    p = mean(q_out)
    m, v = mean_var(q_in)
    h(x) = -p * normlogcdf(x) - (1 - p) * normlogcdf(-x)
    gh = GaussHermiteCubature(points)
    scale = sqrt(2 * v)
    return sum(k -> gh.witer[k] * h(gh.piter[k] * scale + m), 1:points) / sqrt(π)
end

@define_average_energy(
    node = Probit, args = (q[:out]::Union{PointMass, Bernoulli}, q[:in]::UnivariateNormalDistributionsFamily),
    body = (algo, args) -> probit_energy(algo.p, args.q[:out], args.q[:in]),
)

# The rules without expectation propagation, for a model that runs Probit under
# `DefaultAlgorithm()`: the factorisation decides what they read, messages in a joint cluster or
# marginals under mean-field, and the rule towards `in` does not read its own edge.
@define_message_update_rule(
    node = Probit, target = :out, algorithm = DefaultAlgorithm, args = (m[:in]::PointMass,),
    body = (args) -> Bernoulli(normcdf(mean(args.m[:in]))),
)

@define_message_update_rule(
    node = Probit, target = :out, algorithm = DefaultAlgorithm, args = (m[:in]::UnivariateNormalDistributionsFamily,),
    body = (args) -> Bernoulli(normcdf(mean(args.m[:in]) / sqrt(1 + var(args.m[:in])))),
)

@define_message_update_rule(
    node = Probit, target = :out, algorithm = DefaultAlgorithm, args = (q[:in]::PointMass,),
    body = (args) -> Bernoulli(normcdf(mean(args.q[:in]))),
)

# The likelihood of `in` given an output probability p: log(1 - p + (2p - 1) Φ(z)).
probit_likelihood(p) = ContinuousUnivariateLogPdf(z -> log(1 - p + (2 * p - 1) * normcdf(z)))

@define_message_update_rule(
    node = Probit, target = :in, algorithm = DefaultAlgorithm, args = (m[:out]::Union{PointMass, Bernoulli},),
    body = (args) -> probit_likelihood(mean(args.m[:out])),
)

@define_message_update_rule(
    node = Probit, target = :in, algorithm = DefaultAlgorithm, args = (q[:out]::PointMass,),
    body = (args) -> probit_likelihood(mean(args.q[:out])),
)

@define_average_energy(
    node = Probit, algorithm = DefaultAlgorithm, args = (q[:out]::Union{PointMass, Bernoulli}, q[:in]::UnivariateNormalDistributionsFamily),
    body = (args) -> probit_energy(32, args.q[:out], args.q[:in]),
)

end
