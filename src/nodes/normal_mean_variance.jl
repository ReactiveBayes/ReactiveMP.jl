export make_node, rule


function make_node(::Type{ <: NormalMeanVariance }; factorisation = ((1, 2, 3), ))
    return FactorNode(NormalMeanVariance, Stochastic, (:out, :mean, :variance), factorisation, nothing)
end

function make_node(::Type{ <: NormalMeanVariance }, out, mean, variance; factorisation = ((1, 2, 3), ))
    node = make_node(NormalMeanVariance, factorisation = factorisation)
    connect!(node, :out, out)
    connect!(node, :mean, mean)
    connect!(node, :variance, variance)
    return node
end

## rules

function rule(::Type{ <: NormalMeanVariance }, ::Type{ Val{:out} }, ::Marginalisation, messages::Tuple{Message{T}, Message{T}}, ::Nothing, ::Nothing) where { T <: Real }
    return NormalMeanVariance(mean(messages[1]), mean(messages[2]))
end

function rule(::Type{ <: NormalMeanVariance }, ::Type{ Val{:mean} }, ::Marginalisation, messages::Tuple{Message{T}, Message{T}}, ::Nothing, ::Nothing) where { T <: Real }
    return NormalMeanVariance(mean(messages[1]), mean(messages[2]))
end

function rule(::Type{ <: NormalMeanVariance }, ::Type{ Val{:out} }, ::Marginalisation, messages::Tuple{Message{ <: NormalMeanVariance{T} }, Message{T}}, ::Nothing, ::Nothing) where { T <: Real }
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
    return NormalMeanVariance(mean(marginals[1]), mean(marginals[2]))
end

function rule(
    ::Type{ <: NormalMeanVariance }, 
    ::Type{ Val{:out} }, 
    ::Marginalisation, 
    ::Nothing, 
    marginals::Tuple{Marginal, Marginal}, 
    ::Nothing)
    ##
    return NormalMeanVariance(mean(marginals[1]), mean(marginals[2]))
end

function marginalrule(
    ::Type{ <: NormalMeanVariance }, 
    ::Type{ Val{:out_mean_variance} }, 
    messages::Tuple{Message{NormalMeanVariance{Float64}}, Message{Float64},Message{Float64}}, 
    ::Nothing,
    ::Nothing)
    ##
    q_out = Message(NormalMeanVariance(getdata(messages[2]), getdata(messages[3]))) * messages[1]
    return (getdata(q_out), getdata(messages[2]), getdata(messages[3]))
end

function marginalrule(
    ::Type{ <: NormalMeanVariance }, 
    ::Type{ Val{:out_mean_variance} }, 
    messages::Tuple{Message{Float64}, Message{NormalMeanVariance{Float64}}, Message{Float64}}, 
    ::Nothing,
    ::Nothing)
    ##
    q_mean = Message(NormalMeanVariance(getdata(messages[1]), getdata(messages[3]))) * messages[2]
    return (getdata(messages[1]), getdata(q_mean), getdata(messages[3]))
end