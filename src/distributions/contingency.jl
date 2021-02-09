export Contingency

struct Contingency{ T, P <: AbstractMatrix{T} }
    p :: P
end

contingency_matrix(distribution::Contingency) = distribution.p

function entropy(distribution::Contingency)
    P = contingency_matrix(distribution)
    -sum(P .* log.(clamp.(P, tiny, Inf)))
end
