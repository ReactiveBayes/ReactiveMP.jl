export Contingency

using LinearAlgebra

"""
    Contingency(P, renormalize = Val(true))

The contingency distribution is a multivariate generalization of the categorical distribution. As a bivariate distribution, the 
contingency distribution defines the joint probability over two unit vectors `v1` and `v2`. The parameter `P` encodes a contingency matrix that specifies the probability of co-occurrence.

    v1 ∈ {0, 1}^d1 where Σ_j v1_j = 1
    v2 ∈ {0, 1}^d2 where Σ_k v2_k = 1

    P ∈ [0, 1]^{d1 × d2}, where Σ_jk P_jk = 1

    f(v1, v2, P) = Contingency(out1, out2 | P) = Π_jk P_jk^{v1_j * v2_k}

A `Contingency` distribution over more than two variables requires higher-order tensors as parameters; these are not implemented in ReactiveMP.

# Arguments:
- `P`, required, contingency matrix
- `renormalize`, optional, supports either `Val(true)` or `Val(false)`, specifies whether matrix `P` must be automatically renormalized. Does not modify the original `P` and allocates a new one for the renormalized version. If set to `false` the contingency matrix `P` **must** be normalized by hand, otherwise the result of related calculations might be wrong

"""
struct Contingency{T, P <: AbstractMatrix{T}} <: ContinuousMatrixDistribution
    p::P

    Contingency{T, P}(A::AbstractMatrix) where {T, P <: AbstractMatrix{T}} = new(A)
end

Contingency(P::AbstractMatrix)                                               = Contingency(P, Val(true))
Contingency(P::M, renormalize::Val{true}) where {T, M <: AbstractMatrix{T}}  = Contingency{T, M}(P ./ sum(P))
Contingency(P::M, renormalize::Val{false}) where {T, M <: AbstractMatrix{T}} = Contingency{T, M}(P)

contingency_matrix(distribution::Contingency) = distribution.p

vague(::Type{<:Contingency}, dims::Int) = Contingency(ones(dims, dims) ./ abs2(dims))

convert_eltype(::Type{Contingency}, ::Type{T}, distribution::Contingency{R}) where {T <: Real, R <: Real} = Contingency(convert(AbstractArray{T}, contingency_matrix(distribution)))

function entropy(distribution::Contingency)
    P = contingency_matrix(distribution)
    return -mapreduce((p) -> p * clamplog(p), +, P)
end
