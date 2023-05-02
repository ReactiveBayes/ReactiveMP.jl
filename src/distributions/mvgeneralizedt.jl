export MvGeneralizedTDistribution, SamplingMessagesAppproximation, SamplingMessagesAppproximationInstance

using Distributions, Random, SpecialFunctions, StatsFuns, ReactiveMP, Distances, LinearAlgebra

import Base: minimum, maximum

struct MvGeneralizedTDistribution{T <: Real, Cov <: AbstractMatrix, Mean <: AbstractVector} <: ContinuousMultivariateDistribution
    df::T # non-integer degrees of freedom allowed
    dim::Int
    μ::Mean
    Σ::Cov

    function MvGeneralizedTDistribution{T, Cov, Mean}(df::T, dim::Int, μ::Mean, Σ::AbstractMatrix{T}) where {T, Cov, Mean}
        df > zero(df) || error("df must be positive")
        new{T, Cov, Mean}(df, dim, μ, Σ)
    end
end

function MvGeneralizedTDistribution(df::T, μ::Mean, Σ::Cov) where {Cov <: AbstractMatrix, Mean <: AbstractVector, T <: Real}
    d = length(μ)
    size(Σ, 1) == d || throw(DimensionMismatch("The dimensions of μ and Σ are inconsistent."))
    R = Base.promote_eltype(T, μ, Σ)
    S = Base.convert(AbstractArray{R}, Σ)
    m = Base.convert(AbstractArray{R}, μ)
    MvGeneralizedTDistribution{R, typeof(S), typeof(m)}(R(df), d, m, S)
end

function MvGeneralizedTDistribution(df::Real, Σ::AbstractMatrix)
    R = Base.promote_eltype(df, Σ)
    MvGeneralizedTDistribution(df, Zeros{R}(size(Σ, 1)), Σ)
end

MvGeneralizedTDistribution{T, Cov, Mean}(df, μ, Σ) where {T, Cov, Mean} = MvGeneralizedTDistribution(convert(T, df), convert(Mean, μ), convert(Cov, Σ))


struct SamplingMessagesAppproximation{R, U}
    rng::R
    n_samples::Int
    unsafe::Val{U}
end

function SamplingMessagesAppproximation(rng::AbstractRNG, n_samples::Int)
    return SamplingMessagesAppproximation(rng, n_samples, Val(false))
end


# We need a seprate structure to ensure that we create a local cache per node, rather then a global cache
mutable struct SamplingMessagesAppproximationInstance{R, U}
    rng         :: R
    n_samples   :: Int
    unsafe      :: Val{U}
    activated   :: Bool

    outμsamples :: Vector{Vector{Float64}}
    outsamples  :: Vector{Vector{Float64}}
    μsamples    :: Vector{Vector{Float64}}
    νsamples    :: Vector{Float64}
    Σsamples    :: Vector{Matrix{Float64}}
    Σinvsamples :: Vector{Matrix{Float64}}

    tmp1::Vector{Float64}
    tmp2::Matrix{Float64}

    current_qoutμ :: Any
    current_qout  :: Any
    current_qν    :: Any
    current_qμ    :: Any
    current_qΣ    :: Any

    SamplingMessagesAppproximationInstance{R, U}(rng::R, n_samples, unsafe::Val{U}) where {R, U} = new(rng, n_samples, unsafe, false)
end
import Base: copyto! 

Base.copyto!(instance::SamplingMessagesAppproximationInstance, args...) = Base.copyto!(instance, instance.unsafe, args...)
Base.copyto!(instance::SamplingMessagesAppproximationInstance, ::Val{true}, args...) = unsafe_copyto!(args...)
Base.copyto!(instance::SamplingMessagesAppproximationInstance, ::Val{false}, args...) = Base.copyto!(args...)


import Base: convert

### Conversion
function Base.convert(::Type{MvGeneralizedTDistribution{T}}, d::MvGeneralizedTDistribution) where {T <: Real}
    S = Base.convert(AbstractArray{T}, d.Σ)
    m = Base.convert(AbstractArray{T}, d.μ)
    MvGeneralizedTDistribution{T, typeof(S), typeof(m)}(T(d.df), d.dim, m, S)
end

Base.convert(::Type{MvGeneralizedTDistribution{T}}, d::MvGeneralizedTDistribution{T}) where {T <: Real} = d

function Base.convert(::Type{MvGeneralizedTDistribution{T}}, df, dim, μ::AbstractVector, Σ::AbstractMatrix) where {T <: Real}
    S = Base.convert(AbstractArray{T}, Σ)
    m = Base.convert(AbstractArray{T}, μ)
    MvGeneralizedTDistribution{T, typeof(S), typeof(m)}(T(df), dim, m, S)
end

# Basic statistics

Distributions.length(d::MvGeneralizedTDistribution) = d.dim

Distributions.mean(d::MvGeneralizedTDistribution) = d.df > 1 ? d.μ : NaN
Distributions.mode(d::MvGeneralizedTDistribution) = d.μ
Distributions.modes(d::MvGeneralizedTDistribution) = [mode(d)]

Distributions.var(d::MvGeneralizedTDistribution) = d.df > 2 ? (d.df / (d.df - 2)) * diag(d.Σ) : Float64[NaN for i in 1:(d.dim)]
Distributions.scale(d::MvGeneralizedTDistribution) = Matrix(d.Σ)
Distributions.cov(d::MvGeneralizedTDistribution) = d.df > 2 ? (d.df / (d.df - 2)) * Matrix(d.Σ) : NaN * ones(d.dim, d.dim)
Distributions.invscale(d::MvGeneralizedTDistribution) = Matrix(inv(d.Σ))
Distributions.invcov(d::MvGeneralizedTDistribution) = d.df > 2 ? ((d.df - 2) / d.df) * Matrix(inv(d.Σ)) : NaN * ones(d.dim, d.dim)
Distributions.logdet_cov(d::MvGeneralizedTDistribution) = d.df > 2 ? logdet((d.df / (d.df - 2)) * d.Σ) : NaN

Distributions.params(d::MvGeneralizedTDistribution) = (d.df, d.μ, d.Σ)
@inline partype(d::MvGeneralizedTDistribution{T}) where {T} = T
Base.eltype(::Type{<:MvGeneralizedTDistribution{T}}) where {T} = T

##computation is taken from Multivariate-t distributions and applications by Samuel, Kotz pg.23 eqn. 1.29
##for non-central t-distribution we truncate the expansion at 20 order. If more accuracy is needded use big
##float and expand to desired accuracy.
function Distributions.entropy(d::MvGeneralizedTDistribution)
    hdf, hdim = 0.5 * d.df, 0.5 * d.dim
    shdfhdim = hdf + hdim

    # # replace with the below code if more precision is required
    # M = expdelta*mapreduce(i -> (digamma(shdfhdim + i )- digamma(d.df/2))/factorial(big(i)), +, 1:200)
    if all(≈(0), d.m)
        return 0.5 * logdet(d.Σ) + hdim * log(d.df * pi) + logbeta(hdim, hdf) - loggamma(hdim) + shdfhdim * (digamma(shdfhdim) - digamma(hdf))
    else
        delta = sqmahal(d, zeros(d.dim))
        expdelta = exp(-delta / 2)
        M = expdelta * mapreduce(i -> (digamma(shdfhdim + i) - digamma(d.df / 2)) / factorial(i), +, 1:20)
        return 0.5 * logdet(d.Σ) + hdim * log(d.df * pi) + logbeta(hdim, hdf) - loggamma(hdim) + shdfhdim * M
    end
end

Distributions.insupport(d::MvGeneralizedTDistribution, x::AbstractVector{T}) where {T <: Real} = length(d) == length(x) && all(isfinite, x)

Distributions.sqmahal(d::MvGeneralizedTDistribution, x::AbstractVector{<:Real}) = invquad(d.Σ, x - d.μ)

function Distributions.sqmahal!(r::AbstractArray, d::MvGeneralizedTDistribution, x::AbstractMatrix{<:Real})
    invquad!(r, d.Σ, x .- d.μ)
end

Distributions.sqmahal(d::MvGeneralizedTDistribution, x::AbstractMatrix{T}) where {T <: Real} = Distributions.sqmahal!(Vector{T}(undef, size(x, 2)), d, x)

function mvtdist_consts(d::MvGeneralizedTDistribution)
    H = Base.convert(eltype(d), 0.5)
    logpi = Base.convert(eltype(d), log(pi))
    hdf = H * d.df
    hdim = H * d.dim
    shdfhdim = hdf + hdim
    v = loggamma(shdfhdim) - loggamma(hdf) - hdim * log(d.df) - hdim * logpi - H * logdet(d.Σ)
    return (shdfhdim, v)
end

function Distributions._logpdf(d::MvGeneralizedTDistribution, x::AbstractVector{T}) where {T <: Real}
    shdfhdim, v = mvtdist_consts(d)
    v - shdfhdim * log1p(sqmahal(d, x) / d.df)
end

function Distributions._logpdf!(r::AbstractArray, d::MvGeneralizedTDistribution, x::AbstractMatrix)
    sqmahal!(r, d, x)
    shdfhdim, v = mvtdist_consts(d)
    for i in 1:size(x, 2)
        r[i] = v - shdfhdim * log1p(r[i] / d.df)
    end
    return r
end

function Distributions.logpdf(::Type{MvGeneralizedTDistribution}, x::AbstractVector{<:Real}, μ::AbstractVector{<:Real}, Σ::AbstractMatrix{<:Real}, ν::Real)
    return logpdf(MvGeneralizedTDistribution(ν, μ, Σ), x)
end

function Distributions.gradlogpdf(d::MvGeneralizedTDistribution, x::AbstractVector{<:Real})
    z = x - d.μ
    prz = invscale(d) * z
    -((d.df + d.dim) / (d.df + dot(z, prz))) * prz
end

# Sampling (for MvGeneralizedTDistribution)
function Distributions._rand!(rng::AbstractRNG, d::MvGeneralizedTDistribution, x::AbstractVector{<:Real})
    chisqd = Chisq(d.df)
    y = sqrt(rand(rng, chisqd) / d.df)
    unwhiten!(d.Σ, randn!(rng, x))
    x .= x ./ y .+ d.μ
    x
end

function Distributions._rand!(rng::AbstractRNG, d::MvGeneralizedTDistribution, x::AbstractMatrix{T}) where {T <: Real}
    cols = size(x, 2)
    chisqd = Chisq(d.df)
    y = Matrix{T}(undef, 1, cols)
    unwhiten!(d.Σ, randn!(rng, x))
    rand!(rng, chisqd, y)
    x .= x ./ sqrt.(y ./ d.df) .+ d.μ
    x
end

##

using LinearAlgebra, Random, StaticArrays, LoopVectorization

macro check_argdims(cond)
    quote
        ($(esc(cond))) || throw(DimensionMismatch("Inconsistent argument dimensions."))
    end
end
_rcopy!(r, x) = (r === x || copyto!(r, x); r)
whiten!(a::AbstractMatrix, x::AbstractVecOrMat) = whiten!(x, a, x)
unwhiten!(a::AbstractMatrix, x::AbstractVecOrMat) = unwhiten!(x, a, x)

function whiten!(r::AbstractVecOrMat, a::AbstractMatrix, x::AbstractVecOrMat)
    v = _rcopy!(r, x)
    LinearAlgebra.ldiv!(LowerTriangular(ReactiveMP.cholsqrt(a)), v)
end

function unwhiten!(r::AbstractVecOrMat, a::AbstractMatrix, x::AbstractVecOrMat)
    v = _rcopy!(r, x)
    LinearAlgebra.lmul!(LowerTriangular(ReactiveMP.cholsqrt(a)), v)
end

whiten(a::AbstractMatrix, x::AbstractVecOrMat) = whiten!(similar(x), a, x)
unwhiten(a::AbstractMatrix, x::AbstractVecOrMat) = unwhiten!(similar(x), a, x)

invquad(a::AbstractMatrix, x::AbstractVector) = sum(abs2, LowerTriangular(ReactiveMP.cholsqrt(a)) \ x)

invquad(a::AbstractMatrix, x::AbstractVecOrMat) = x' / a * x
function invquad(a::AbstractMatrix{T}, x::AbstractMatrix{S}) where {T <: Real, S <: Real}
    @check_argdims LinearAlgebra.checksquare(a) == size(x, 1)
    invquad!(Array{promote_type(T, S)}(undef, size(x, 2)), a, x)
end

invquad!(r::AbstractArray, a::AbstractMatrix, x::AbstractMatrix) = colwise_dot!(r, x, a \ x)