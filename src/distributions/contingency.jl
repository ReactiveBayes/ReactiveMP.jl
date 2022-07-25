export Contingency

using LinearAlgebra

struct Contingency{T, P <: AbstractMatrix{T}} <: ContinuousMatrixDistribution
    p::P
end

contingency_matrix(distribution::Contingency) = distribution.p

vague(::Type{<:Contingency}, dims::Int) = Contingency(ones(dims, dims) ./ abs2(dims))

function entropy(distribution::Contingency)
    P = contingency_matrix(distribution)
    return -mapreduce((p) -> p * clamplog(p), +, P)
end
