export make_node, rule

function NormalMeanPrecisionNode(::Type{T} = Float64; factorisation = ((1, 2, 3), )) where T
    return FactorNode(NormalMeanPrecision{T}, Stochastic, (:mean, :precision, :value), factorisation)
end

function make_node(::Type{ <: NormalMeanPrecision{T} }; factorisation = ((1, 2, 3), )) where T
    return NormalMeanPrecisionNode(T, factorisation = factorisation)
end

function make_node(::Type{ <: NormalMeanPrecision{T} }, mean, precision, value; factorisation = ((1, 2, 3), )) where T
    node = make_node(NormalMeanPrecision{T}, factorisation = factorisation)
    connect!(node, :mean, mean)
    connect!(node, :precision, precision)
    connect!(node, :value, value)
    return node
end

## rules

function rule(::Type{ <: NormalMeanPrecision{T} }, ::Type{ Val{:value} }, ::Marginalisation, messages::Tuple{Message{T}, Message{T}}, marginals::Nothing, meta) where { T <: Real }
    return NormalMeanPrecision{T}(mean(messages[1]), mean(messages[2]))
end

function rule(::Type{ <: NormalMeanPrecision{T} }, ::Type{ Val{:mean} }, ::Marginalisation, ::Nothing, marginals::Tuple{Marginal, Marginal}, meta) where { T <: Real }
    return NormalMeanPrecision{T}(mean(marginals[2]), mean(marginals[1]))
end

function rule(::Type{ <: NormalMeanPrecision{T} }, ::Type{ Val{:precision} }, ::Marginalisation, ::Nothing, marginals::Tuple{Marginal, Marginal}, meta) where { T <: Real }
    diff = mean(marginals[2]) - mean(marginals[1])
    return GammaAB{T}(3.0 / 2.0, 1.0 / 2.0 * (var(marginals[1]) + var(marginals[2]) + diff^2))
end

function rule(::Type{ <: NormalMeanPrecision{T} }, ::Type{ Val{:value} }, ::Marginalisation, ::Nothing, marginals::Tuple{Marginal, Marginal}, meta) where { T <: Real }
    return NormalMeanPrecision{T}(mean(marginals[1]), mean(marginals[2]))
end

## marginal rules

function marginalrule(::Type{ <: NormalMeanPrecision{T} }, ::Type{ Val{ :mean_precision_value } }, messages::Tuple{Message{T}, Message{T}, Message{NormalMeanPrecision{T}}}, ::Nothing, ::Nothing) where { T <: Real }
    q_value = Message(NormalMeanPrecision(getdata(messages[1]), getdata(messages[2]))) * messages[3]
    return (getdata(messages[1]), getdata(messages[2]), getdata(q_value))
end