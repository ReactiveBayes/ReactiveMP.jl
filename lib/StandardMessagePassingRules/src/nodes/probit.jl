"""
    Probit

The probit node, `out ~ Bernoulli(Φ(in))` with `Φ` the standard normal CDF: a binary output
through a real latent input. It is stochastic, with the interfaces

- `out`: the output, observed as a `PointMass` at `0` or `1`, or
  a `Bernoulli`; its value, the probability of a `1`, lies in `[0, 1]`;
- `in`: the latent input, a univariate normal.

Its algorithm is [`ProbitEP`](@ref), and it also has rules under
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm). Under `DefaultAlgorithm` the
message towards `in` is the exact likelihood, as a log-density, rather than a normal. Both
algorithms have an average energy, computed by
[`GaussHermiteCubature`](@extref MessagePassingRulesApproximations.GaussHermiteCubature). No rule
declares a log scale.

Under `ProbitEP` the rule towards `in` reads the message on `in` itself, so in a graph with a
loop through `in` it needs one to start from: the node declares
`NormalMeanPrecision(0.0, 100.0)`, set at activation wherever the model sets no initial message.

# Throws

- [`RuleInputError`](@extref MessagePassingRulesBase.RuleInputError), from the
  expectation-propagation rules towards `in` and the joint marginal, when the value on `out`
  lies outside `[0, 1]`.

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
  `32`, the number the average energy under `DefaultAlgorithm` uses too. The messages are
  computed in closed form and do not use it.

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

# The check of the expectation-propagation rules: the value on `out` is a probability.
check_probability(m_out) = (p = mean(m_out); zero(p) <= p <= one(p) || lazy"the value on `out` must lie in [0, 1]; got $p")

# The moments of the tilted distribution Φ-likelihood × the cavity m_in, for an output
# probability p in [0, 1], and the variance clamped to the cavity's.
function tilted_moments(p, m_in)
    mz, vz = mean_cov(m_in)
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

# Under `DefaultAlgorithm()`, with the same cubature at 32 points.
@define_average_energy(
    node = Probit, algorithm = DefaultAlgorithm, args = (q[:out]::Union{PointMass, Bernoulli}, q[:in]::UnivariateNormalDistributionsFamily),
    body = (args) -> probit_energy(32, args.q[:out], args.q[:in]),
)
