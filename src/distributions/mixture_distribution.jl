import Distributions: components, component

export MixtureDistribution
export components, component, weights

"""

    `MixtureDistribution{C,CT<:Real}`

    A custom mixture distribution implementation, parameterized by:
    * `C` type family of the mixture
    * `CT` the type for the weights

    This implementation solves:
    * [Distributions.jl Issue 1669](https://github.com/JuliaStats/Distributions.jl/issues/1669)
    * [ReactiveMP.jl Issue 253](https://github.com/biaslab/ReactiveMP.jl/issues/253)

"""
struct MixtureDistribution{C, CT <: Real}
    components::Vector{C}
    weights::Vector{CT}

    function MixtureDistribution(cs::Vector{C}, w::Vector{CT}) where {C, CT}
        length(cs) == length(w) || error("The number of components does not match the length of prior.")
        @assert all(>=(0), w) "weight vector contains negative entries."
        @assert sum(w) == 1 "weight vector is not normalized."
        new{C, CT}(cs, w)
    end
end

"""
    components(dist::MixtureDistribution)

Returns the components of the mixture distribution `dist`.
"""
components(dist::MixtureDistribution) = dist.components

"""
    component(dist::MixtureDistribution, k::Int)

Returns the `k`'th component of the mixture distribution `dist`.
"""
component(dist::MixtureDistribution, k::Int) = dist.components[k]

"""
    weights(dist::MixtureDistribution)

Returns the weights of the mixture distribution `dist`.
"""
weights(dist::MixtureDistribution) = dist.weights

"""
    mean(dist::MixtureDistribution)

Returns the mean of the mixture distribution `dist`.
"""
mean(dist::MixtureDistribution) = dot(weights(dist), mean.(components(dist)))

"""
    var(dist::MixtureDistribution)

Returns the variance of the mixture distribution `dist`.
"""
function var(dist::MixtureDistribution)
    w = weights(dist)
    dists = components(dist)
    result = 0.0
    for k in 1:length(w)
        result += w[k] * (var(dists[k]) + mean(dists[k])^2)
    end
    return result - mean(dist)^2
end

"""
    prod(::ProdAnalytical, left::MixtureDistribution, right::Any)

Computes the analytical product between a `MixtureDistribution` and something else.
"""
function prod(::ProdAnalytical, left::MixtureDistribution, right::Any)

    # get weights and components
    w = weights(left)
    dists = components(left)

    # get new distributions
    dists_new = map(dist -> prod(ProdAnalytical(), dist, right), dists)

    # get scales
    logscales = map((dist, dist_new) -> prod(AddonProdLogScale(), dist_new, dist, right), dists, dists_new)

    # compute updated weights
    logscales_new = log.(w) + logscales

    # return mixture distribution
    return MixtureDistribution(dists_new, softmax(logscales_new))
end

"""
    prod(::ProdAnalytical, left::Any, right::MixtureDistribution)

Computes the analytical product between a `MixtureDistribution` and something else.
"""
prod(::ProdAnalytical, left::Any, right::MixtureDistribution) = prod(ProdAnalytical(), right, left)

function prod(::AddonProdLogScale, new_dist::MixtureDistribution, left_dist::MixtureDistribution, right_dist::Any)

    # get prior weights and components
    w = left_dist.weights
    dists = left_dist.components

    # get new distributions
    dists_new = map(dist -> prod(ProdAnalytical(), dist, right_dist), dists)

    # get scales
    logscales = map((dist, dist_new) -> prod(AddonProdLogScale(), dist_new, dist, right_dist), dists, dists_new)

    # compute updated weights
    logscales_new = log.(w) + logscales

    return logsumexp(logscales_new)
end

prod(::AddonProdLogScale, new_dist::MixtureDistribution, left_dist::Any, right_dist::MixtureDistribution) = prod(AddonProdLogScale(), new_dist, right_dist, left_dist)
