export Unscented

struct Unscented <: AbstractNonLinearApproximation
    L  :: Int64
    α  :: Float64
    β  :: Float64
    κ  :: Float64
    λ  :: Float64
    Wm :: Vector{Float64}
    Wc :: Vector{Float64}
end

function Unscented(dim::Int64; α::Float64 = 1e-3, β::Float64 = 2.0, κ::Float64 = 0.0)
    λ = α^2 * (dim + κ) - dim
    Wm = ones(2 * dim + 1)
    Wc = ones(2 * dim + 1)
    Wm ./= (2 * (dim + λ))
    Wc ./= (2 * (dim + λ))
    Wm[1] = λ / (dim + λ)
    Wc[1] = λ / (dim + λ) + (1 - α^2 + β)
    return Unscented(dim, α, β, κ, λ, Wm, Wc)
end

# get-functions for the Unscented structure
getL(approximation::Unscented)  = approximation.L
getα(approximation::Unscented)  = approximation.α
getβ(approximation::Unscented)  = approximation.β
getκ(approximation::Unscented)  = approximation.κ
getλ(approximation::Unscented)  = approximation.λ
getWm(approximation::Unscented) = approximation.Wm
getWc(approximation::Unscented) = approximation.Wc
