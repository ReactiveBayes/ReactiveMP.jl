export make_node, rule

function NormalMeanVarianceNode(::Type{T} = Float64; factorisation = ((1, 2, 3), )) where T
    return FactorNode(NormalMeanVariance{T}, Stochastic, (:mean, :variance, :value), factorisation)
end

function make_node(::Type{ <: NormalMeanVariance{T} }; factorisation = ((1, 2, 3), )) where T
    return NormalMeanVarianceNode(T, factorisation = factorisation)
end

function make_node(::Type{ <: NormalMeanVariance{T} }, mean, variance, value; factorisation = ((1, 2, 3), )) where T
    node = make_node(NormalMeanVariance{T}, factorisation = factorisation)
    connect!(node, :mean, mean)
    connect!(node, :variance, variance)
    connect!(node, :value, value)
    return node
end

## rules

function rule(::Type{ <: NormalMeanVariance }, ::Type{ Val{:value} }, ::Marginalisation, messages::Tuple{Message{T}, Message{T}}, ::Nothing, meta) where { T <: Real }
    return NormalMeanVariance(mean(messages[1]), mean(messages[2]))
end

function rule(::Type{ <: NormalMeanVariance }, ::Type{ Val{:mean} }, ::Marginalisation, messages::Tuple{Message{T}, Message{T}}, ::Nothing, meta) where { T <: Real }
    return NormalMeanVariance(mean(messages[2]), mean(messages[1]))
end

function rule(::Type{ <: NormalMeanVariance }, ::Type{ Val{:value} }, ::Marginalisation, messages::Tuple{Message{ <: NormalMeanVariance{T} }, Message{T}}, ::Nothing, meta) where { T <: Real }
    return NormalMeanVariance(mean(messages[1]), var(messages[1]) + mean(messages[2]))
end