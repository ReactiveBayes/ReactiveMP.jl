export GaussianMeanVarianceNode, GaussianMeanPrecisionNode

import Distributions: Normal

## GaussianMeanVariance

function GaussianMeanVarianceNode(::Type{T} = Float64; factorisation = SA[ SA[ 1, 2, 3 ] ]) where T
    return Node(Normal{T}, SA[ :mean, :variance, :value ], factorisation)
end

# Messages ordered as Tuple{ :mean, :variance }
# BP rule
function rule(::Type{ <: Normal{T} }, ::Val{:value}, ::Marginalisation, messages::Tuple{Message{T}, Message{T}}, beliefs::Nothing, meta) where { T <: Real }
    return Normal{T}(mean(messages[1]), sqrt(mean(messages[2])))
end

## GaussianMeanPrecision

function GaussianMeanPrecisionNode(::Type{T} = Float64; factorisation = SA[ SA[ 1, 2, 3 ] ]) where T
    return Node(NormalMeanPrecision{T}, SA[ :mean, :precision, :value ], factorisation)
end

function rule(::Type{ <: NormalMeanPrecision{T} }, ::Val{:value}, ::Marginalisation, messages::Tuple{Message{T}, Message{T}}, beliefs::Nothing, meta) where { T <: Real }
    return NormalMeanPrecision{T}(mean(messages[1]), mean(messages[2]))
end

function rule(::Type{ <: NormalMeanPrecision{T} }, ::Val{:mean}, ::Marginalisation, ::Nothing, beliefs::Tuple{Belief, Belief}, meta) where { T <: Real }
    return NormalMeanPrecision{T}(mean(beliefs[2]), mean(beliefs[1]))
end

function rule(::Type{ <: NormalMeanPrecision{T} }, ::Val{:precision}, ::Marginalisation, ::Nothing, beliefs::Tuple{Belief, Belief}, meta) where { T <: Real }
    diff = mean(beliefs[2]) - mean(beliefs[1])
    return GammaAB{T}(3.0 / 2.0, 1.0 / 2.0 * (var(beliefs[1]) + var(beliefs[2]) + diff^2))
end

function rule(::Type{ <: NormalMeanPrecision{T} }, ::Val{:value}, ::Marginalisation, ::Nothing, beliefs::Tuple{Belief, Belief}, meta) where { T <: Real }
    return NormalMeanPrecision{T}(mean(beliefs[1]), mean(beliefs[2]))
end
