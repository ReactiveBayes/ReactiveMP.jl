"""
    MultinomialPolya

A multinomial through logistic stick-breaking, `x ~ Multinomial(N, p(ψ))` with
`p_k = σ(ψ_k) Π_{j<k} (1 - σ(ψ_j))` for `K` categories and `K - 1` weights `ψ`, Pólya-Gamma
augmented so that the message towards `ψ` is normal. Its interfaces are `x`, the counts, `N`,
the number of trials, and `ψ`. Can be used for multinomial regression.

Its rules run under its own algorithm, [`MultinomialPolyaApproximation`](@ref). The rule towards
`ψ` reads the message on `ψ` itself, beside the marginals the factorisation gives, so a model
initialises that message.
"""
struct MultinomialPolya end

"""
    MultinomialPolyaApproximation(; points = 21)

[`MultinomialPolya`](@ref)'s algorithm: its average energy computes `⟨softplus(ψ_k)⟩` by
Gauss–Hermite cubature with `points` points.
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
    logistic_stick_breaking(m)

The `K` probabilities of logistic stick-breaking with the `K - 1` weights `m`:
`p_k = σ(m_k) Π_{j<k} (1 - σ(m_j))`, and the rest of the stick for the last.

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
    compose_Nks(x, N)

The number of trials left at each of the first `K - 1` breaks of the stick for the counts `x`
out of `N`: `N_k = N - Σ_{j<k} x_j`.

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
