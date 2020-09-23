export make_node, rule


function make_node(::Type{ <: GammaAB }; factorisation = ((1, 2, 3), ))
    return FactorNode(GammaAB, Stochastic, (:out, :a, :b), factorisation, nothing)
end

function make_node(::Type{ <: GammaAB }, out, a, b; factorisation = ((1, 2, 3), ))
    node = make_node(GammaAB, factorisation = factorisation)
    connect!(node, :out, out)
    connect!(node, :a, a)
    connect!(node, :b, b)
    return node
end

