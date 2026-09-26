"""
    MultinomialPolya

The stochastic node of a multinomial through logistic stick-breaking,

```math
p(x \\mid N, \\psi) = \\frac{N!}{\\prod_{k=1}^{K} x_k!} \\prod_{k=1}^{K} p_k^{x_k}, \\qquad
p_k = \\sigma(\\psi_k) \\prod_{j<k} \\big(1 - \\sigma(\\psi_j)\\big), \\quad
p_K = \\prod_{j<K} \\big(1 - \\sigma(\\psi_j)\\big),
```

for `K` categories and `K - 1` log-odds `ψ`, Pólya-Gamma augmented so that the message towards
`ψ` is normal. The multinomial factors into `K - 1` binomials, the `k`-th of `x_k` out of the
`N_k = N - Σ_{j<k} x_j` trials left, with log-odds `ψ_k`. With `ψ` a linear function of
covariates it is multinomial regression.

# Interfaces

- `x`: the counts of the `K` categories, a vector: observed (`PointMass`) or a `Multinomial`;
- `N`: the number of trials, a `PointMass`, or a `Poisson`, `Binomial` or `Categorical`, whose
  mode the rules use;
- `ψ`: the `K - 1` log-odds, `MvNormal`, or a univariate `Normal` for `K = 2`.

# Augmentation

$(DOC_POLYA_AUGMENTATION)

Applied to each binomial of the stick with `b = N_k`, the message towards `ψ` has the weighted
mean `x_k - N_k / 2` and the diagonal precision `ω_k`, the Pólya-Gamma mean at the `k`-th entry of
the mean of the message on `ψ`. The message towards `x` is the multinomial at
[`logistic_stick_breaking`](@ref) of the mean of `q(ψ)`.

The average energy is that of the multinomial itself,
`-⟨log C(x)⟩ - Σ_{k<K} ⟨x_k⟩⟨ψ_k⟩ + Σ_{k<K} N_k ⟨softplus(ψ_k)⟩`, with each `⟨softplus(ψ_k)⟩`
by Gauss–Hermite cubature over the mean and variance of `ψ_k`, with the `points` of
[`MultinomialPolyaApproximation`](@ref). For a `Multinomial` `q(x)`, the coefficient's
expectation is `log N! - Σ_k ⟨log x_k!⟩` over its binomial marginals.

# Limitations

- The rule towards `ψ` reads the **message on `ψ`** itself, besides the marginals its
  factorisation gives it, so a model must initialise that message.
- There is no rule towards `N`, and the average energy takes a `PointMass` `q(N)` only.
- The rules use the mean of `ψ` only: its variance enters the average energy, not the messages.

# Examples

```jldoctest; setup = :(using PolyaMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily)
julia> result = @call_message_update_rule(
           node = MultinomialPolya, target = :ψ,
           q = (x = PointMass([2, 3, 5]), N = PointMass(10)),
           m = (ψ = MvNormalWeightedMeanPrecision(zeros(2), [1.0 0.0; 0.0 1.0]),),
       );

julia> weightedmean(getresult(result)) ≈ [-3.0, -1.0]
true

julia> precision(getresult(result)) ≈ [2.5 0.0; 0.0 2.0]
true
```

The stick's two breaks leave `N_k = 10` and `8` trials, so the weighted means are `2 - 5` and
`3 - 4` and the precisions, at `ψ = 0`, `N_k / 4`.

See also [`MultinomialPolyaApproximation`](@ref), [`logistic_stick_breaking`](@ref),
[`compose_Nks`](@ref), [`BinomialPolya`](@ref).
"""
struct MultinomialPolya end

"""
    MultinomialPolyaApproximation(; points = 21)

The algorithm of [`MultinomialPolya`](@ref), and its default: a model names it only to change
`points`.

# Keywords

- `points`: the number of Gauss–Hermite points with which the average energy computes each
  `⟨softplus(ψ_k)⟩`. Default `21`. The messages do not depend on it.

# Examples

```jldoctest; setup = :(using PolyaMessagePassingRules)
julia> MultinomialPolyaApproximation().points
21
```
"""
struct MultinomialPolyaApproximation <: AbstractAlgorithm
    points::Int
end

MultinomialPolyaApproximation(; points = 21) = MultinomialPolyaApproximation(points)

@define_factor_node(
    node = MultinomialPolya, type = Stochastic, interfaces = [:x, :N, :ψ], algorithm = MultinomialPolyaApproximation,
    dependencies = [:x => (default,), :N => (default,), :ψ => (default, m[:ψ])],
)

"""
    logistic_stick_breaking(m) -> Vector{Float64}

The `K` probabilities of logistic stick-breaking with the `K - 1` log-odds `m`:
`p_k = σ(m_k) Π_{j<k} (1 - σ(m_j))` for `k < K`, and the rest of the stick, `Π_{j<K} (1 - σ(m_j))`,
for the last. The probabilities of [`MultinomialPolya`](@ref) at `ψ = m`.

# Examples

```jldoctest
julia> using PolyaMessagePassingRules

julia> logistic_stick_breaking([0.0, 0.0])
3-element Vector{Float64}:
 0.5
 0.25
 0.25
```
"""
function logistic_stick_breaking(m)
    Km1 = length(m)
    p = Array{Float64}(undef, Km1 + 1)
    remaining = 1.0
    @inbounds for i in 1:Km1
        v = logistic(m[i])
        p[i] = v * remaining
        remaining *= (1 - v)
    end
    p[end] = remaining
    return p
end

"""
    compose_Nks(x, N) -> Vector

The number of trials left at each of the first `K - 1` breaks of the stick, for the `K` counts
`x` out of `N`: `N_k = N - Σ_{j<k} x_j`, of the element type of `x`. The trials of the binomials
[`MultinomialPolya`](@ref) factors into.

# Examples

```jldoctest
julia> using PolyaMessagePassingRules

julia> compose_Nks([2, 3, 5], 10)
2-element Vector{Int64}:
 10
  8
```
"""
function compose_Nks(x, N)
    T = eltype(x)
    K = length(x)
    Nks = Vector{T}(undef, K - 1)
    prev_sum = zero(T)
    @inbounds for k in 1:(K - 1)
        Nks[k] = N - prev_sum
        if k < K - 1
            prev_sum += x[k]
        end
    end
    return Nks
end

# Towards `ψ`: each break's binomial, Pólya-Gamma augmented at the mean of the message on `ψ`.
@define_message_update_rule(
    node = MultinomialPolya, target = :ψ,
    args = (q[:x]::Any, q[:N]::Union{PointMass, Poisson, Binomial, Categorical}, m[:ψ]::NormalDistributionsFamily),
    body = (args) -> begin
        x = mean(args.q[:x])
        K = length(x)
        Nks = compose_Nks(x, mode(args.q[:N]))
        ω = map((n, c) -> mean(PolyaGammaHybridSampler(n, c)), Nks, mean(args.m[:ψ]))
        η = map((d, n) -> d - n / 2, view(x, 1:(K - 1)), Nks)
        Λ = length(η) == 1 ? ω[1] : Diagonal(ω)
        η = length(η) == 1 ? η[1] : η
        convert(promote_variate_type(typeof(η), NormalWeightedMeanPrecision), η, Λ)
    end,
)

# Towards `x`: the multinomial at the stick-breaking probabilities of the mean of `ψ`.
@define_message_update_rule(
    node = MultinomialPolya, target = :x,
    args = (q[:N]::Union{PointMass, Poisson, Binomial, Categorical}, q[:ψ]::Any),
    body = (args) -> Multinomial(mode(args.q[:N]), logistic_stick_breaking(mean(args.q[:ψ]))),
)

# E[log x_k!] for a binomial marginal of `q(x)`.
expected_log_gamma(binomial) = mapreduce(k -> loggamma(k + 1) * pdf(binomial, k), +, 0:ntrials(binomial))

# ⟨-log p(x | N, ψ)⟩ = -⟨log C(x)⟩ - Σ_{k<K} ⟨x_k⟩⟨ψ_k⟩ + Σ_{k<K} ⟨N_k⟩⟨softplus(ψ_k)⟩, the
# coefficient C(x) = N! / Π_k x_k! of the multinomial, and each ⟨N_k⟩ linear in ⟨x⟩.
function multinomial_polya_energy(algo, q_x, q_N, q_ψ)
    N = mean(q_N)
    x = mean(q_x)
    K = length(x)
    Nks = compose_Nks(x, N)
    gh = ghcubature(algo.points)
    expectation(m, v) = sum(((w, p),) -> w * softplus(p), zip(getweights(gh, m, v), getpoints(gh, m, v)))
    expectations = map(expectation, mean(q_ψ), var(q_ψ))
    # For observed counts, log C is the sum of each break's binomial coefficient. For a
    # Multinomial q(x), ⟨log C⟩ = log N! - Σ_k ⟨log x_k!⟩ over the binomial marginals.
    log_coefficient = if q_x isa PointMass
        mapreduce((Nk, y) -> loggamma(Nk + 1) - loggamma(Nk - y + 1) - loggamma(y + 1), +, Nks, x)
    else
        loggamma(N + 1) - sum(p -> expected_log_gamma(Binomial(N, p)), probs(q_x))
    end
    return -log_coefficient - sum(x[1:(K - 1)] .* mean(q_ψ)) + mapreduce(*, +, expectations, Nks)
end

@define_average_energy(
    node = MultinomialPolya,
    args = (q[:x]::Union{PointMass, Multinomial}, q[:N]::PointMass, q[:ψ]::Union{NormalDistributionsFamily, PointMass}),
    body = (algo, args) -> multinomial_polya_energy(algo, args.q[:x], args.q[:N], args.q[:ψ]),
)
