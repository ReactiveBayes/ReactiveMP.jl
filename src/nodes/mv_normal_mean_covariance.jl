export make_node, rule

function make_node(::Type{ <: MvNormalMeanCovariance }; factorisation = ((1, 2, 3), ))
    return FactorNode(MvNormalMeanCovariance, Stochastic, (:out, :mean, :variance), factorisation, nothing)
end

function make_node(::Type{ <: MvNormalMeanCovariance }, out, mean, variance; factorisation = ((1, 2, 3), ))
    node = make_node(MvNormalMeanCovariance, factorisation = factorisation)
    connect!(node, :out, out)
    connect!(node, :mean, mean)
    connect!(node, :variance, variance)
    return node
end

## rules

function rule(
    ::Type{ <: MvNormalMeanCovariance }, 
    ::Type{ <: Union{ Val{:out}, Val{:mean} } }, 
    ::Marginalisation, 
    messages::Tuple{ Message{ <: Dirac }, Message{ <: Dirac } }, 
    ::Nothing, 
    ::Nothing)
    ##
    return MvNormalMeanCovariance(mean(messages[1]), mean(messages[2]))
end

function rule(
    ::Type{ <: MvNormalMeanCovariance }, 
    ::Type{ Val{:out} }, 
    ::Marginalisation, 
    ::Nothing, 
    marginals::Tuple{ Marginal{ <: MvNormalMeanCovariance }, Marginal{ <: Dirac } }, 
    ::Nothing)
    ##
    return MvNormalMeanCovariance(mean(marginals[1]), mean(marginals[2]))
end

function rule(
    ::Type{ <: MvNormalMeanCovariance }, 
    ::Type{ Val{:mean} }, 
    ::Marginalisation, 
    ::Nothing, 
    marginals::Tuple{ Marginal{ <: MvNormalMeanCovariance }, Marginal{ <: Dirac } }, 
    ::Nothing)
    ##
    return MvNormalMeanCovariance(mean(marginals[1]), mean(marginals[2]))
end

function rule(
    ::Type{ <: MvNormalMeanCovariance }, 
    ::Type{ Val{:out} }, 
    ::Marginalisation, 
    messages::Tuple{ Message{ <: MvNormalMeanCovariance } }, 
    marginals::Tuple{ Marginal{ <: Dirac } }, 
    ::Nothing)
    ##
    return MvNormalMeanCovariance(mean(messages[1]), cov(messages[1]) + mean(marginals[1]))
end

function rule(
    ::Type{ <: MvNormalMeanCovariance }, 
    ::Type{ Val{:mean} }, 
    ::Marginalisation, 
    messages::Tuple{ Message{ <: MvNormalMeanCovariance } }, 
    marginals::Tuple{ Marginal{ <: Dirac } }, 
    ::Nothing)
    ##
    return MvNormalMeanCovariance(mean(messages[1]), cov(messages[1]) + mean(marginals[1]))
end