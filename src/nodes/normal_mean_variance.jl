export make_node, rule


function make_node(::Type{ <: NormalMeanVariance }; factorisation = ((1, 2, 3), ))
    return FactorNode(NormalMeanVariance, Stochastic, (:out, :μ, :v), factorisation, nothing)
end

function make_node(::Type{ <: NormalMeanVariance }, out, μ, v; factorisation = ((1, 2, 3), ))
    node = make_node(NormalMeanVariance, factorisation = factorisation)
    connect!(node, :out, out)
    connect!(node, :μ, μ)
    connect!(node, :v, v)
    return node
end