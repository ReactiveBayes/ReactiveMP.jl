export make_node, rule

function MvNormalMeanCovarianceNode(::Type{T} = Float64; factorisation = ((1, 2, 3), )) where T
    return FactorNode(MvNormalMeanCovariance{T}, Stochastic, (:mean, :variance, :value), factorisation, nothing)
end

function make_node(::Type{ <: MvNormalMeanCovariance{T} }; factorisation = ((1, 2, 3), )) where T
    return MvNormalMeanCovarianceNode(T, factorisation = factorisation)
end

function make_node(::Type{ <: MvNormalMeanCovariance{T} }, mean, variance, value; factorisation = ((1, 2, 3), )) where T
    node = make_node(MvNormalMeanCovariance{T}, factorisation = factorisation)
    connect!(node, :mean, mean)
    connect!(node, :variance, variance)
    connect!(node, :value, value)
    return node
end

## rules

function rule(
    ::Type{ <: MvNormalMeanCovariance }, 
    ::Type{ Val{:value} }, 
    ::Marginalisation, 
    messages::Tuple{Message{Vector{T}}, Message{ <: PDMat{T} }}, 
    ::Nothing, 
    ::Nothing) where { T <: Real }
    ##
    return MvNormalMeanCovariance(getdata(messages[1]), getdata(messages[2]))
end

function rule(
    ::Type{ <: MvNormalMeanCovariance }, 
    ::Type{ Val{:mean} }, 
    ::Marginalisation, 
    messages::Tuple{Message{ <: PDMat{T} }, Message{Vector{T}}}, 
    ::Nothing, 
    ::Nothing) where { T <: Real }
    ##
    return MvNormalMeanCovariance(getdata(messages[2]), getdata(messages[1]))
end

function rule(
    ::Type{ <: MvNormalMeanCovariance }, 
    ::Type{ Val{:value} }, 
    ::Marginalisation, 
    ::Nothing, 
    marginals::Tuple{Marginal{MvNormalMeanCovariance{T}}, Marginal{ <: PDMat{T} }}, 
    ::Nothing) where { T <: Real }
    ##
    return MvNormalMeanCovariance(mean(marginals[1]), getdata(marginals[2]))
end

function rule(
    ::Type{ <: MvNormalMeanCovariance }, 
    ::Type{ Val{:mean} }, 
    ::Marginalisation, 
    ::Nothing, 
    marginals::Tuple{ Marginal{ <: PDMat{T} }, Marginal{MvNormalMeanCovariance{T}}}, 
    ::Nothing) where { T <: Real }
    ##
    return MvNormalMeanCovariance(mean(marginals[2]), getdata(marginals[1]))
end

function rule(
    ::Type{ <: MvNormalMeanCovariance }, 
    ::Type{ Val{:value} }, 
    ::Marginalisation, 
    messages::Tuple{Message{ <: MvNormalMeanCovariance{T} }}, 
    marginals::Tuple{Marginal{ <: PDMat{T} }}, 
    ::Nothing) where { T <: Real }
    ##
    return MvNormalMeanCovariance(mean(messages[1]), cov(messages[1]) + getdata(marginals[1]))
end

function rule(
    ::Type{ <: MvNormalMeanCovariance }, 
    ::Type{ Val{:mean} }, 
    ::Marginalisation, 
    messages::Tuple{Message{ <: MvNormalMeanCovariance{T} }}, 
    marginals::Tuple{Marginal{ <: PDMat{T} }}, 
    ::Nothing) where { T <: Real }
    ##
    return MvNormalMeanCovariance(mean(messages[1]), cov(messages[1]) + getdata(marginals[1]))
end