export GaussianMeanVarianceNode, GaussianMeanPrecisionNode

import Distributions: Normal

## GaussianMeanVariance

function GaussianMeanVarianceNode(::Type{T} = Float64; factorisation = ((1, 2, 3), )) where T
    return FactorNode(Normal{T}, (:mean, :variance, :value), factorisation)
end

# Messages ordered as Tuple{ :mean, :variance }
# BP rule
function rule(::Type{ <: Normal{T} }, ::Val{:value}, ::Marginalisation, messages::Tuple{Message{T}, Message{T}}, marginals::Nothing, meta) where { T <: Real }
    return Normal{T}(mean(messages[1]), sqrt(mean(messages[2])))
end

## GaussianMeanPrecision

function GaussianMeanPrecisionNode(::Type{T} = Float64; factorisation = ((1, 2, 3), )) where T
    return FactorNode(NormalMeanPrecision{T}, (:mean, :precision, :value), factorisation)
end

function rule(::Type{ <: NormalMeanPrecision{T} }, ::Val{:value}, ::Marginalisation, messages::Tuple{Message{T}, Message{T}}, marginals::Nothing, meta) where { T <: Real }
    return NormalMeanPrecision{T}(mean(messages[1]), mean(messages[2]))
end

function rule(::Type{ <: NormalMeanPrecision{T} }, ::Val{:mean}, ::Marginalisation, ::Nothing, marginals::Tuple{Marginal, Marginal}, meta) where { T <: Real }
    return NormalMeanPrecision{T}(mean(marginals[2]), mean(marginals[1]))
end

function rule(::Type{ <: NormalMeanPrecision{T} }, ::Val{:precision}, ::Marginalisation, ::Nothing, marginals::Tuple{Marginal, Marginal}, meta) where { T <: Real }
    diff = mean(marginals[2]) - mean(marginals[1])
    return GammaAB{T}(3.0 / 2.0, 1.0 / 2.0 * (var(marginals[1]) + var(marginals[2]) + diff^2))
end

function rule(::Type{ <: NormalMeanPrecision{T} }, ::Val{:value}, ::Marginalisation, ::Nothing, marginals::Tuple{Marginal, Marginal}, meta) where { T <: Real }
    return NormalMeanPrecision{T}(mean(marginals[1]), mean(marginals[2])) # VMP rule here
end
