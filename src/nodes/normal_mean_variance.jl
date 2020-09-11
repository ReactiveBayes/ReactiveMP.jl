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