export Categorical
export CategoricalNaturalParameters

import Distributions: Categorical, probs

vague(::Type{<:Categorical}, dims::Int) = Categorical(ones(dims) ./ dims)

convert_paramfloattype(::Type{T}, dist::Categorical) where {T} = Categorical(convert_paramfloattype(T, probs(dist)))

prod_analytical_rule(::Type{<:Categorical}, ::Type{<:Categorical}) = ProdAnalyticalRuleAvailable()

function prod(::ProdAnalytical, left::Categorical, right::Categorical)
    # Multiplication of 2 categorical PMFs: p(z) = p(x) * p(y)
    mvec = clamp.(probvec(left) .* probvec(right), tiny, huge)
    norm = sum(mvec)
    return Categorical(mvec ./ norm)
end

probvec(dist::Categorical) = probs(dist)

function compute_logscale(new_dist::Categorical, left_dist::Categorical, right_dist::Categorical)
    return log(dot(probvec(left_dist), probvec(right_dist)))
end

struct CategoricalNaturalParameters{T <: Real, M <: AbstractArray{T}} <: NaturalParameters
    η::M
end

function Base.convert(::Type{CategoricalNaturalParameters}, dist::Categorical)
    p = probvec(dist)
    η = log.(p / p[end])
    return CategoricalNaturalParameters(η)
end

Base.convert(::Type{CategoricalNaturalParameters}, vec::AbstractVector) = convert(CategoricalNaturalParameters{eltype(vec)}, vec)

Base.convert(::Type{CategoricalNaturalParameters{T}}, vec::AbstractVector) where {T} = CategoricalNaturalParameters(convert(AbstractVector{T}, vec))

as_naturalparams(::Type{T}, args...) where {T <: CategoricalNaturalParameters} = convert(CategoricalNaturalParameters, args...)

function Base.convert(::Type{Distribution}, params::CategoricalNaturalParameters)
    return Categorical(softmax(params.η))
end

function Base.:(==)(left::CategoricalNaturalParameters, right::CategoricalNaturalParameters)
    return left.η == right.η
end

function isproper(::CategoricalNaturalParameters)
    return true
end

function lognormalizer(params::CategoricalNaturalParameters)
    return log(sum(exp.(params.η)))
end

logpdf(params::CategoricalNaturalParameters, x) = params.η[x] - lognormalizer(params)