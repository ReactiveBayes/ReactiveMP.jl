export make_node, rule


function make_node(::Type{ <: NormalMeanPrecision }; factorisation = ((1, 2, 3), ))
    return FactorNode(NormalMeanPrecision, Stochastic, (:out, :mean, :precision), factorisation, nothing)
end

function make_node(::Type{ <: NormalMeanPrecision }, out, mean, precision; factorisation = ((1, 2, 3), ))
    node = make_node(NormalMeanPrecision, factorisation = factorisation)
    connect!(node, :out, out)
    connect!(node, :mean, mean)
    connect!(node, :precision, precision)
    return node
end

## rules

function rule(
    ::Type{ <: NormalMeanPrecision }, 
    ::Type{ <: Union{ Val{:out}, Val{:mean} } }, 
    ::Marginalisation, 
    messages::Tuple{ Message{ <: Dirac{T} }, Message{ <: Dirac{T} } }, 
    ::Nothing, 
    ::Nothing) where T
    return NormalMeanPrecision(mean(messages[1]), mean(messages[2]))
end

function rule(
    ::Type{ <: NormalMeanPrecision }, 
    ::Type{ <: Union{ Val{:out}, Val{:mean} } }, 
    ::Marginalisation, 
    ::Nothing, 
    marginals::Tuple{Marginal, Marginal}, 
    ::Nothing)
    ##
    return NormalMeanPrecision(mean(marginals[1]), mean(marginals[2]))
end

function rule(
    ::Type{ <: NormalMeanPrecision }, 
    ::Type{ Val{:precision} }, 
    ::Marginalisation, 
    ::Nothing, 
    marginals::Tuple{Marginal, Marginal}, 
    ::Nothing)
    ##
    diff = mean(marginals[1]) - mean(marginals[2])
    return GammaAB(3.0 / 2.0, 1.0 / 2.0 * (var(marginals[2]) + var(marginals[1]) + diff^2))
end

function rule(
    ::Type{ <: NormalMeanPrecision }, 
    ::Type{ Val{:out} }, 
    ::Marginalisation, 
    ::Nothing, 
    marginals::Tuple{Marginal, Marginal}, 
    ::Nothing)
    ##
    return NormalMeanPrecision(mean(marginals[1]), mean(marginals[2]))
end

## marginal rules

function marginalrule(
    ::Type{ <: NormalMeanPrecision }, 
    ::Type{ Val{ :out_mean_precision } }, 
    messages::Tuple{ Message{ <: NormalMeanPrecision{T} }, Message{ <: Dirac{T} }, Message{ <: Dirac{T} } }, 
    ::Nothing, 
    ::Nothing) where T
    q_out = Message(NormalMeanPrecision(mean(messages[2]), mean(messages[3]))) * messages[1]
    return (getdata(q_out), getdata(messages[2]), getdata(messages[3]))
end