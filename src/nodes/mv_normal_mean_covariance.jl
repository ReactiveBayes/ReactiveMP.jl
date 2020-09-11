export make_node, rule

function make_node(::Type{ <: MvNormalMeanCovariance }; factorisation = ((1, 2, 3), ))
    return FactorNode(MvNormalMeanCovariance, Stochastic, (:out, :mean, :covariance), factorisation, nothing)
end

function make_node(::Type{ <: MvNormalMeanCovariance }, out, mean, covariance; factorisation = ((1, 2, 3), ))
    node = make_node(MvNormalMeanCovariance, factorisation = factorisation)
    connect!(node, :out, out)
    connect!(node, :mean, mean)
    connect!(node, :covariance, covariance)
    return node
end