@doc raw"""
    BIFM

The deterministic node of a whole time slice of a linear state-space model, for
backward-information-filter forward-marginal (BIFM) smoothing:

```math
z_{next} = A z_{prev} + B u, \qquad out = C z_{next},
```

with the matrices `A`, `B` and `C` carried by its algorithm, [`BIFMSmoother`](@ref), which the
model must give each BIFM node. A chain of BIFM nodes starts with [`BIFMHelper`](@ref).

# Interfaces

- `out`: the output, `C znext`, a vector;
- `in`: the input `u`, a vector;
- `zprev`: the previous state;
- `znext`: the next state.

Every message the rules read is a multivariate normal; the forward rules read the message on
`zprev` as a `TerminalProdArgument` wrapping one, the marginal that [`BIFMHelper`](@ref) or the
previous slice sends.

# Smoothing

Smoothing runs in two passes over the chain. The backward pass sends information-form messages
towards `zprev`, from the messages on `out`, `in` and `znext`:

```math
\begin{aligned}
\xi_z &= C^\top \xi_{out} + \xi_{znext}, &
\Lambda_z &= C^\top \Lambda_{out} C + \Lambda_{znext}, &
H &= (\Lambda_{in} + B^\top \Lambda_z B)^{-1}, \\
\tilde\xi_z &= \xi_z + \Lambda_z B H (-\xi_{in} - B^\top \xi_z), &
\tilde\Lambda_z &= \Lambda_z (I - B H B^\top \Lambda_z), &
m_{zprev} &= \mathcal{N}^{-1}(A^\top \tilde\xi_z,\; A^\top \tilde\Lambda_z A).
\end{aligned}
```

[`BIFMHelper`](@ref) turns it into the forward pass, whose messages towards `in`, `out` and
`znext` are the marginals of those variables themselves, as `TerminalProdArgument`s, so that the
engine takes them as the marginals without a product. Each forward rule reads the message on its
own edge, besides the others, and recomputes the backward quantities from them: the rules keep
nothing between calls, so they are independent of the order they run in, and one
[`BIFMSmoother`](@ref) may serve every node. On a chain the result is the Rauch–Tung–Striebel
smoother's.

# Limitations

- **No free energy.** A deterministic node's free energy reads the joint marginal of its
  inputs, `q(in, zprev, znext)`, whose rule here throws a
  [`BIFMFreeEnergyError`](@ref BIFMMessagePassingRules.BIFMFreeEnergyError), as does the average
  energy of [`BIFMHelper`](@ref). Run the inference without the free energy.
- Multivariate normal messages only, with a `TerminalProdArgument` on `zprev` for the forward
  rules: the chain must start with [`BIFMHelper`](@ref).
- Under the default algorithm,
  [`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm), no rule exists: every
  node must be given a [`BIFMSmoother`](@ref).

# Examples

A scalar random walk observed without transformation: the message on `out` is `N(1, 1)`, the
input's `N(0, 1)`, and nothing is known of the next state yet, so the backward message on `zprev`
is `N(1, 2)`.

```jldoctest; setup = :(using BIFMMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily)
julia> algorithm = BIFMSmoother([1.0;;], [1.0;;], [1.0;;]);

julia> result = @call_message_update_rule(
           node = BIFM, target = :zprev, algorithm = algorithm,
           m = (
               out = MvNormalMeanPrecision([1.0], [1.0;;]),
               in = MvNormalMeanPrecision([0.0], [1.0;;]),
               znext = MvNormalWeightedMeanPrecision([0.0], [0.0;;]),
           ),
       );

julia> m, V = mean_cov(getresult(result));

julia> m ≈ [1.0] && V ≈ [2.0;;]
true
```

In a model, with RxInfer, the chain starts at a [`BIFMHelper`](@ref) and ends at a flat prior
on the last state:

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

See also [`BIFMSmoother`](@ref), [`BIFMHelper`](@ref).
"""
struct BIFM end

"""
    BIFMSmoother(A, B, C)

The algorithm of [`BIFM`](@ref): the matrices of `znext = A zprev + B in` and `out = C znext`,
converted to a common element type. It keeps nothing between calls, so one smoother may serve
every node of a chain; the statistics of the input come from the message on `in`. The node
declares no algorithm of its own, so a model must give this one.

# Arguments

- `A`: the `d × d` state transition;
- `B`: the `d × k` input matrix, for a `k`-dimensional input;
- `C`: the `p × d` output matrix, for a `p`-dimensional output.

# Throws

A `DimensionMismatch` when `A` is not square, `B` has not as many rows as `A`, or `C` has not
as many columns as `A` has rows.

# Examples

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

# Each forward rule also reads the message on its own edge, to recompute the backward quantities.
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
# `scratch`, in this order of operations:
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
    BIFMFreeEnergyError(node::Symbol)

The error thrown when the free energy of a model with [`BIFM`](@ref) or [`BIFMHelper`](@ref)
is asked for, which is not supported: BIFM smoothing computes the posteriors, and its Bethe free
energy is not derived. `node` names the node that threw, `:BIFM` or `:BIFMHelper`.
"""
struct BIFMFreeEnergyError <: Exception
    node::Symbol
end

Base.showerror(io::IO, e::BIFMFreeEnergyError) = print(
    io, "the free energy of a model with `$(e.node)` is not supported: BIFM smoothing computes the posteriors, ",
    "and its Bethe free energy is not derived. Run the inference without the free energy.",
)

# A deterministic node's free energy reads the marginal of its inputs' joint, and BIFM's is where
# the free energy of a BIFM model is asked for, so this rule throws.
@define_marginal_update_rule(
    node = BIFM, target = (:in, :zprev, :znext), algorithm = BIFMSmoother,
    args = (m[:out]::Any, m[:in]::Any, m[:zprev]::Any, m[:znext]::Any),
    body = (args) -> throw(BIFMFreeEnergyError(:BIFM)),
)
