export make_node, rule

function make_node(::Type{ <: MvNormalMeanCovariance }; factorisation = ((1, 2, 3), ))
    return FactorNode(MvNormalMeanCovariance, Stochastic, (:out, :μ, :Σ), factorisation, nothing)
end

function make_node(::Type{ <: MvNormalMeanCovariance }, out, μ, Σ; factorisation = ((1, 2, 3), ))
    node = make_node(MvNormalMeanCovariance, factorisation = factorisation)
    connect!(node, :out, out)
    connect!(node, :μ, μ)
    connect!(node, :Σ, Σ)
    return node
end