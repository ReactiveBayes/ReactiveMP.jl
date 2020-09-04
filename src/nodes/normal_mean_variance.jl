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

function rule(
    ::Type{ <: NormalMeanVariance }, 
    ::Type{ Val{:mean} }, 
    ::Marginalisation, 
    ::Nothing, 
    marginals::Tuple{Marginal,Marginal}, 
    ::Nothing)
    ##
    return NormalMeanVariance(mean(marginals[2]), mean(marginals[1]))
end

function rule(
    ::Type{ <: NormalMeanVariance }, 
    ::Type{ Val{:value} }, 
    ::Marginalisation, 
    ::Nothing, 
    marginals::Tuple{Marginal, Marginal}, 
    ::Nothing)
    ##
    return NormalMeanVariance(mean(marginals[1]), mean(marginals[2]))
end

function marginalrule(
    ::Type{ <: NormalMeanVariance }, 
    ::Type{ Val{:mean_variance_value} }, 
    messages::Tuple{Message{Float64},Message{Float64},Message{NormalMeanVariance{Float64}}}, 
    ::Nothing,
    ::Nothing)
    ##
    q_out = Message(NormalMeanVariance(getdata(messages[1]), getdata(messages[2]))) * messages[3]
    return (getdata(messages[1]), getdata(messages[2]), getdata(q_out))
end

function marginalrule(
    ::Type{ <: NormalMeanVariance }, 
    ::Type{ Val{:mean_variance_value} }, 
    messages::Tuple{Message{NormalMeanVariance{Float64}},Message{Float64},Message{Float64}}, 
    ::Nothing,
    ::Nothing)
    ##
    q_mean = Message(NormalMeanVariance(getdata(messages[3]), getdata(messages[2]))) * messages[1]
    return (getdata(q_mean), getdata(messages[2]), getdata(messages[3]))
end