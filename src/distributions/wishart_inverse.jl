export InverseWishart

import Distributions: InverseWishart, Wishart, pdf!
import Base: ndims, size, convert
import LinearAlgebra
import StatsFuns: logπ, logmvgamma
import SpecialFunctions: digamma, loggamma

"""
    InverseWishartMessage

Same as `InverseWishart` from `Distributions.jl`, but does not check input arguments and allows creating improper `InverseWishart` message.
For model creation use `InverseWishart` from `Distributions.jl`. Regular user should never interact with `InverseWishartMessage`.
"""
struct InverseWishartMessage{T <: Real, A <: AbstractMatrix{T}} <: ContinuousMatrixDistribution
    ν::T
    S::A
end

function InverseWishartMessage(ν::Real, S::AbstractMatrix{<:Real})
    T = promote_type(typeof(ν), eltype(S))
    return InverseWishartMessage(convert(T, ν), convert(AbstractArray{T}, S))
end

InverseWishartMessage(ν::Integer, S::AbstractMatrix{Real}) = InverseWishartMessage(float(ν), S)

Distributions.params(dist::InverseWishartMessage) = (dist.ν, dist.S)
Distributions.mean(dist::InverseWishartMessage)   = mean(convert(InverseWishart, dist))
Distributions.var(dist::InverseWishartMessage)    = var(convert(InverseWishart, dist))
Distributions.cov(dist::InverseWishartMessage)    = cov(convert(InverseWishart, dist))
Distributions.mode(dist::InverseWishartMessage)   = mode(convert(InverseWishart, dist))

mean_cov(dist::InverseWishartMessage) = mean_cov(convert(InverseWishart, dist))

Base.size(dist::InverseWishartMessage)           = size(dist.S)
Base.size(dist::InverseWishartMessage, dim::Int) = size(dist.S, dim)

const InverseWishartDistributionsFamily{T} = Union{InverseWishart{T}, InverseWishartMessage{T}}

to_marginal(dist::InverseWishartMessage) = convert(InverseWishart, dist)

function Base.convert(::Type{InverseWishartMessage{T}}, distribution::InverseWishartMessage) where {T}
    (ν, S) = params(distribution)
    return InverseWishartMessage(convert(T, ν), convert(AbstractMatrix{T}, S))
end

# from "Parametric Bayesian Estimation of Differential Entropy and Relative Entropy" Gupta et al.
function Distributions.entropy(dist::InverseWishartDistributionsFamily)
    d = size(dist, 1)
    ν, S = params(dist)
    d * (d - 1) / 4 * logπ + mapreduce(i -> loggamma((ν + 1.0 - i) / 2), +, 1:d) + ν / 2 * d + (d + 1) / 2 * (logdet(S) - log(2)) -
    (ν + d + 1) / 2 * mapreduce(i -> digamma((ν - d + i) / 2), +, 1:d)
end

function Distributions.mean(::typeof(logdet), dist::InverseWishartDistributionsFamily)
    d = size(dist, 1)
    ν, S = params(dist)
    return -(mapreduce(i -> digamma((ν + 1 - i) / 2), +, 1:d) + d * log(2) - logdet(S))
end

function Distributions.mean(::typeof(inv), dist::InverseWishartDistributionsFamily)
    return mean(cholinv, dist)
end

function Distributions.mean(::typeof(cholinv), dist::InverseWishartDistributionsFamily)
    ν, S = params(dist)
    return mean(Wishart(ν, cholinv(S)))
end

samplefloattype(sampleable::InverseWishartMessage) = promote_type(typeof(sampleable.ν), eltype(sampleable.S))
sampletype(sampleable::InverseWishartMessage) = Matrix{samplefloattype(sampleable)}

function Distributions.rand(rng::AbstractRNG, sampleable::InverseWishartMessage)
    container = [zeros(samplefloattype(sampleable), size(sampleable))]
    rand!(rng, sampleable, container)
    return first(container)
end

function Distributions.rand!(rng::AbstractRNG, sampleable::InverseWishartMessage, x::AbstractVector{<:AbstractMatrix})
    (df, S⁻¹) = Distributions.params(sampleable)
    S = cholinv(S⁻¹)
    L = Distributions.PDMats.chol_lower(ReactiveMP.fastcholesky(S))

    # check 
    p = size(S, 1)
    singular = df <= p - 1
    if singular
        isinteger(df) || throw(ArgumentError("df of a singular Wishart distribution must be an integer (got $df)"))
    end

    A     = similar(S)
    l     = length(S)
    axes2 = axes(A, 2)

    for C in x

        # Wishart sample
        if singular
            r = rank(S)
            randn!(rng, view(A, :, axes2[1:r]))
            fill!(view(A, :, axes2[(r + 1):end]), zero(eltype(A)))
        else
            Distributions._wishart_genA!(rng, A, df)
        end
        # Distributions.unwhiten!(S, A)
        lmul!(L, A)

        M = Cholesky(A, 'L', convert(LinearAlgebra.BlasInt, 0))
        LinearAlgebra.inv!(M)

        copyto!(C, 1, M.L.data, 1, l)
    end

    return x
end

function Distributions.pdf!(out::AbstractArray{<:Real}, distribution::ReactiveMP.InverseWishartMessage, samples::AbstractArray{<:AbstractMatrix{<:Real}, O}) where {O}
    @assert length(out) === length(samples) "Invalid dimensions in pdf!"

    p = size(distribution, 1)
    (df, Ψ) = Distributions.params(distribution)

    T = copy(Ψ)
    R = similar(T)
    l = length(T)

    M = ReactiveMP.fastcholesky!(T)

    # logc0 evaluation
    h_df = df / 2
    Ψld = logdet(M)
    logc0 = -h_df * (p * convert(typeof(df), Distributions.logtwo) - Ψld) - logmvgamma(p, h_df)

    # logkernel evaluation 
    @inbounds for i in 1:length(out)
        copyto!(T, 1, samples[i], 1, l)
        C = ReactiveMP.fastcholesky!(T)
        ld = logdet(C)
        LinearAlgebra.inv!(C)
        mul!(R, Ψ, C.factors)
        r = tr(R)
        out[i] = exp(-0.5 * ((df + p + 1) * ld + r) + logc0)
    end

    return out
end

vague(::Type{<:InverseWishart}, dims::Integer) = InverseWishart(dims + 2, tiny .* diageye(dims))

Base.ndims(dist::InverseWishart) = size(dist, 1)

function Base.convert(::Type{InverseWishart}, dist::InverseWishartMessage)
    (ν, S) = params(dist)
    return InverseWishart(ν, Matrix(Hermitian(S)))
end

Base.convert(::Type{InverseWishartMessage}, dist::InverseWishart) = InverseWishartMessage(params(dist)...)

## Utility functions

convert_paramfloattype(::Type{T}, dist::InverseWishart) where {T} = InverseWishart(convert_paramfloattype.(T, params(dist))...)
convert_paramfloattype(::Type{T}, dist::InverseWishartMessage) where {T} = InverseWishartMessage(convert_paramfloattype(T, dist.ν), convert_paramfloattype(T, dist.S))

## Friendly functions

function logpdf_sample_friendly(dist::InverseWishartMessage)
    return (dist, dist)
end

# We do not define prod between `InverseWishart` from `Distributions.jl` for a reason
# We want to compute `prod` only for `InverseWishartMessage` messages as they are significantly faster in creation
prod_analytical_rule(::Type{<:InverseWishartMessage}, ::Type{<:InverseWishartMessage}) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::InverseWishartMessage, right::InverseWishartMessage)
    @assert size(left, 1) === size(right, 1) "Cannot compute a product of two InverseWishart distributions of different sizes"

    d = size(left, 1)

    ldf, lS = params(left)
    rdf, rS = params(right)

    V = lS + rS

    df = ldf + rdf + d + 1

    return InverseWishartMessage(df, V)
end
