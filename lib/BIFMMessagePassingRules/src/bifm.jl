@doc raw"""
    BIFM

A whole time slice of a linear state-space model, for backward-information-filter
forward-marginal (BIFM) smoothing:

```math
z_{next} = A z_{prev} + B u, \qquad out = C z_{next},
```

with `in` the input `u`. Its interfaces are `out`, `in`, `zprev` and `znext`. It is used with
[`BIFMHelper`](@ref) at the start of the chain, and its rules run under [`BIFMSmoother`](@ref),
which the model gives each BIFM node:

```julia
z_prior ~ MvNormalMeanPrecision(zeros(2), diageye(2))
z[1] ~ BIFMHelper(z_prior)
for i in eachindex(y)
    u[i] ~ MvNormalMeanPrecision(μu, Wu)
    yt[i] ~ BIFM(u[i], z[i], z[i + 1])        # under BIFMSmoother(A, B, C)
    y[i] ~ MvNormalMeanPrecision(yt[i], Wy)
end
z[end] ~ MvNormalMeanPrecision(zeros(2), zeros(2, 2))
```

with `z_prior` and `z` in separate clusters, `q(z_prior) q(z)`.

The backward pass sends information-form messages towards `zprev`; the forward pass returns
`TerminalProdArgument`s, the marginals of `in`, `out` and `znext` themselves. Each forward rule
reads the message on its own edge, as well as the others, and computes the backward quantities
from them, so the rules are pure and independent of the order they run in. The free energy of
a model with BIFM is not supported.
"""
struct BIFM end

"""
    BIFMSmoother(A, B, C)

[`BIFM`](@ref)'s algorithm: the transition matrices of `znext = A zprev + B in` and
`out = C znext`. v6 called it `BIFMMeta(A, B, C)`, a mutable cache its rules shared; the port
keeps nothing between calls, and `BIFMMeta`'s five-argument form, which fixed the input's
statistics, has no counterpart, since the message on `in` gives them.

```jldoctest
julia> using BIFMMessagePassingRules

julia> BIFMSmoother([1.0 0.0; 0.0 1.0], reshape([1.0, 0.0], 2, 1), [1.0 0.0]).C
1×2 Matrix{Float64}:
 1.0  0.0
```
"""
struct BIFMSmoother{T} <: AbstractAlgorithm
    A::Matrix{T}
    B::Matrix{T}
    C::Matrix{T}
    function BIFMSmoother(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)
        size(A, 1) == size(A, 2) || throw(DimensionMismatch("`A` must be square, got $(size(A))"))
        size(A, 1) == size(B, 1) || throw(DimensionMismatch("`B` must have as many rows as `A`, $(size(A, 1)); got $(size(B))"))
        size(A, 1) == size(C, 2) || throw(DimensionMismatch("`C` must have as many columns as `A` has rows, $(size(A, 1)); got $(size(C))"))
        T = promote_type(eltype(A), eltype(B), eltype(C))
        return new{T}(convert(Matrix{T}, A), convert(Matrix{T}, B), convert(Matrix{T}, C))
    end
end

@define_factor_node(node = BIFM, type = Deterministic, interfaces = [:out, :in, :zprev, :znext])

# Each forward rule also reads the message on its own edge, where v6 read the cache the rule
# towards `zprev` had filled.
@define_dependencies(
    node = BIFM, algorithm = BIFMSmoother,
    dependencies = [:out => (default, m[:out]), :in => (default, m[:in]), :zprev => (default,), :znext => (default, m[:znext])],
)

# The rules' working memory, of the dimensions of the state and the output and of the promoted
# float type of the algorithm and the messages.
function bifm_scratch(algo::BIFMSmoother, messages...)
    T = promote_type(eltype(algo.A), promote_samplefloattype(messages...))
    dz, du, dy = size(algo.A, 1), size(algo.B, 2), size(algo.C, 1)
    return (
        ξz = zeros(T, dz), Λz = zeros(T, dz, dz), BHBt = zeros(T, dz, dz), ξ̃z = zeros(T, dz), Λ̃z = zeros(T, dz, dz),
        CtΛ = zeros(T, dz, dy), BH = zeros(T, dz, du), z = zeros(T, dz), zz = zeros(T, dz, dz),
    )
end

# The backward quantities, computed from the messages on `out`, `in` and `znext` into
# `scratch`, in v6's order of operations:
#
#     ξz = Cᵀξ_out + ξ_znext,      Λz = CᵀΛ_out C + Λ_znext,      H = (Λ_in + BᵀΛz B)⁻¹,
#     ξ̃z = ξz + Λz B H (-ξ_in - Bᵀξz),      Λ̃z = Λz (I - B H Bᵀ Λz).
#
# Returns `scratch`; every field is written before it is read.
function bifm_backward!(scratch, algo::BIFMSmoother, m_out, m_in, m_znext)
    A, B, C = algo.A, algo.B, algo.C
    ξ_in, Λ_in = weightedmean_precision(m_in)
    ξ_out, Λ_out = weightedmean_precision(m_out)
    ξ_znext, Λ_znext = weightedmean_precision(m_znext)

    mul!(scratch.ξz, C', ξ_out)
    scratch.ξz .+= ξ_znext
    mul!(scratch.CtΛ, C', Λ_out)
    mul!(scratch.Λz, scratch.CtΛ, C)
    scratch.Λz .+= Λ_znext

    H = cholinv(Λ_in + B' * scratch.Λz * B)
    mul!(scratch.BH, B, H)
    mul!(scratch.BHBt, scratch.BH, B')

    mul!(scratch.z, B, H * (-ξ_in - B' * scratch.ξz))
    mul!(scratch.ξ̃z, scratch.Λz, scratch.z)
    scratch.ξ̃z .+= scratch.ξz

    mul!(scratch.zz, scratch.BHBt, scratch.Λz)
    scratch.zz .= I - scratch.zz
    mul!(scratch.Λ̃z, scratch.Λz, scratch.zz)
    return scratch
end

# Towards `zprev`, the backward pass: an information-form message.
@define_message_update_rule(
    node = BIFM, target = :zprev, algorithm = BIFMSmoother,
    args = (m[:out]::MultivariateNormalDistributionsFamily, m[:in]::MultivariateNormalDistributionsFamily, m[:znext]::MultivariateNormalDistributionsFamily),
    scratch = (algo, args) -> bifm_scratch(algo, args.m[:out], args.m[:in], args.m[:znext]),
    body = (scratch, algo, args) -> begin
        s = bifm_backward!(scratch, algo, args.m[:out], args.m[:in], args.m[:znext])
        A = algo.A
        T = promote_samplefloattype(args.m[:out], args.m[:in], args.m[:znext])
        convert(MvNormalWeightedMeanPrecision{T}, MvNormalWeightedMeanPrecision(A' * s.ξ̃z, A' * s.Λ̃z * A))
    end,
)

# Towards `in`, the forward pass: the marginal of the input, from its message and the marginal of
# `zprev` (Wadehn 2016, *On sparsity by NUV-EM*, in dual parameterisation).
@define_message_update_rule(
    node = BIFM, target = :in, algorithm = BIFMSmoother,
    args = (
        m[:out]::MultivariateNormalDistributionsFamily, m[:in]::MultivariateNormalDistributionsFamily,
        m[:zprev]::TerminalProdArgument{<:MultivariateNormalDistributionsFamily}, m[:znext]::MultivariateNormalDistributionsFamily,
    ),
    scratch = (algo, args) -> bifm_scratch(algo, args.m[:out], args.m[:in], args.m[:znext]),
    body = (scratch, algo, args) -> begin
        s = bifm_backward!(scratch, algo, args.m[:out], args.m[:in], args.m[:znext])
        A, B = algo.A, algo.B
        μu, Σu = mean_cov(args.m[:in])
        μ_zprev, Σ_zprev = mean_cov(args.m[:zprev].argument)
        ξ̃x = s.Λ̃z * (A * μ_zprev) - s.ξ̃z
        Λ̃x = s.Λ̃z - s.Λ̃z * A * Σ_zprev * A' * s.Λ̃z
        BΣu = B * Σu
        μ_in = μu - (Σu * (B' * ξ̃x))
        Σ_in = Σu - BΣu' * Λ̃x * BΣu
        T = promote_samplefloattype(args.m[:out], args.m[:in], args.m[:zprev].argument, args.m[:znext])
        TerminalProdArgument(convert(MvNormalMeanCovariance{T}, MvNormalMeanCovariance(μ_in, Σ_in)))
    end,
)

# The forward statistics of `znext`, shared by the rules towards `out` and `znext`.
function bifm_forward(s, algo::BIFMSmoother, m_in, m_zprev)
    A, B = algo.A, algo.B
    μ_in, Σ_in = mean_cov(m_in)
    μ_zprev, Σ_zprev = mean_cov(m_zprev.argument)
    F = I - s.Λz * s.BHBt
    μ_znext = F' * (A * μ_zprev) + B * ((Σ_in * (B' * s.ξ̃z)) + μ_in)
    Σ_znext = F' * (A * Σ_zprev * A') * F + s.BHBt
    return μ_znext, Σ_znext
end

@define_message_update_rule(
    node = BIFM, target = :out, algorithm = BIFMSmoother,
    args = (
        m[:out]::MultivariateNormalDistributionsFamily, m[:in]::MultivariateNormalDistributionsFamily,
        m[:zprev]::TerminalProdArgument{<:MultivariateNormalDistributionsFamily}, m[:znext]::MultivariateNormalDistributionsFamily,
    ),
    scratch = (algo, args) -> bifm_scratch(algo, args.m[:out], args.m[:in], args.m[:znext]),
    body = (scratch, algo, args) -> begin
        s = bifm_backward!(scratch, algo, args.m[:out], args.m[:in], args.m[:znext])
        μ_znext, Σ_znext = bifm_forward(s, algo, args.m[:in], args.m[:zprev])
        C = algo.C
        T = promote_samplefloattype(args.m[:out], args.m[:in], args.m[:zprev].argument, args.m[:znext])
        TerminalProdArgument(convert(MvNormalMeanCovariance{T}, MvNormalMeanCovariance(C * μ_znext, C * Σ_znext * C')))
    end,
)

@define_message_update_rule(
    node = BIFM, target = :znext, algorithm = BIFMSmoother,
    args = (
        m[:out]::MultivariateNormalDistributionsFamily, m[:in]::MultivariateNormalDistributionsFamily,
        m[:zprev]::TerminalProdArgument{<:MultivariateNormalDistributionsFamily}, m[:znext]::MultivariateNormalDistributionsFamily,
    ),
    scratch = (algo, args) -> bifm_scratch(algo, args.m[:out], args.m[:in], args.m[:znext]),
    body = (scratch, algo, args) -> begin
        s = bifm_backward!(scratch, algo, args.m[:out], args.m[:in], args.m[:znext])
        μ_znext, Σ_znext = bifm_forward(s, algo, args.m[:in], args.m[:zprev])
        T = promote_samplefloattype(args.m[:out], args.m[:in], args.m[:zprev].argument, args.m[:znext])
        TerminalProdArgument(convert(MvNormalMeanCovariance{T}, MvNormalMeanCovariance(μ_znext, Σ_znext)))
    end,
)

"""
    BIFMFreeEnergyError(node)

The free energy of a model with [`BIFM`](@ref) or [`BIFMHelper`](@ref) was asked for, which is
not supported: v6 could not compute it either.
"""
struct BIFMFreeEnergyError <: Exception
    node::Symbol
end

Base.showerror(io::IO, e::BIFMFreeEnergyError) = print(
    io, "the free energy of a model with `$(e.node)` is not supported: BIFM smoothing computes the posteriors, ",
    "and its Bethe free energy is not derived. Run the inference without the free energy.",
)

# A deterministic node's free energy reads the marginal of its inputs' joint, and BIFM's is where
# the free energy of a BIFM model is asked for. v6's rule returned the joint of `in` and `zprev`
# only, for a cluster over all three, and is not ported.
@define_marginal_update_rule(
    node = BIFM, target = (:in, :zprev, :znext), algorithm = BIFMSmoother,
    args = (m[:out]::Any, m[:in]::Any, m[:zprev]::Any, m[:znext]::Any),
    body = (args) -> throw(BIFMFreeEnergyError(:BIFM)),
)
