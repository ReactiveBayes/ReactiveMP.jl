export make_node, rule

import Distributions: Wishart

function make_node(::Type{ <: Wishart }; factorisation = ((1, 2, 3), ))
    return FactorNode(Wishart, Stochastic, (:out, :ν, :S), factorisation, nothing)
end

function make_node(::Type{ <: Wishart }, out, ν, S; factorisation = ((1, 2, 3), ))
    node = make_node(Wishart, factorisation = factorisation)
    connect!(node, :out, out)
    connect!(node, :ν, ν)
    connect!(node, :S, S)
    return node
end
