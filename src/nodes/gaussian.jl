export GaussianMeanVarianceNode, GaussianMeanPrecisionNode

import Distributions: Normal

## GaussianMeanVariance

function GaussianMeanVarianceNode(::Type{T} = Float64; factorisation = SA[ SA[ 1, 2, 3 ] ]) where T
    return Node(Normal{T}, SA[ :mean, :variance, :value ], factorisation)
end

# Messages ordered as Tuple{ :mean, :variance }
# BP rule
function rule(::Type{ <: Normal{T}}, ::Val{:value}, ::Marginalisation, messages::Tuple{Message{T}, Message{T}}, beliefs::Nothing, meta) where { T <: Real }
    mean   = getdata(messages[1])
    stddev = sqrt(getdata(messages[2]))
    return Message(Normal{T}(mean, stddev))
end

## GaussianMeanPrecision

function GaussianMeanPrecisionNode(::Type{T} = Float64; factorisation = SA[ SA[ 1, 2, 3 ] ]) where T
    return Node(NormalMeanPrecision{T}, SA[ :mean, :precision, :value ], factorisation)
end

function rule(::Type{ <: NormalMeanPrecision{T} }, ::Val{:value}, ::Marginalisation, messages::Tuple{Message{T}, Message{T}}, beliefs::Nothing, meta) where { T <: Real }
    mean      = getdata(messages[1])
    precision = getdata(messages[2])
    return Message(NormalMeanPrecision{T}(mean, precision))
end

function rule(::Type{ <: NormalMeanPrecision{T} }, ::Val{:mean}, ::Marginalisation, messages::Nothing, beliefs::Tuple{Belief{Gamma{T}}, Belief{NormalMeanPrecision{T}}}, meta) where { T <: Real }
    precision_belief = getdata(beliefs[1])
    value_belief     = getdata(beliefs[2])
    return Message(NormalMeanPrecision{T}(mean(value_belief), mean(precision_belief)))
end

function rule(::Type{ <: NormalMeanPrecision{T} }, ::Val{:precision}, ::Marginalisation, messages::Nothing, beliefs::Tuple{Belief{NormalMeanPrecision{T}}, Belief{NormalMeanPrecision{T}}}, meta) where { T <: Real }
    mean_belief  = getdata(beliefs[1])
    value_belief = getdata(beliefs[2])
    dif = mean(value_belief) - mean(mean_belief)
    return Message(Gamma{T}(3.0 / 2.0, 1.0 / 2.0 * (var(mean_belief) + var(value_belief) + dif^2)))
end

function rule(::Type{ <: NormalMeanPrecision{T} }, ::Val{:value}, ::Marginalisation, messages::Nothing, beliefs::Tuple{Belief{NormalMeanPrecision{T}}, Belief{Gamma{T}}}, meta) where { T <: Real }
    mean_belief      = getdata(beliefs[1])
    precision_belief = getdata(beliefs[2])
    return Message(NormalMeanPrecision{T}(mean(mean_belief), mean(precision_belief)))
end
