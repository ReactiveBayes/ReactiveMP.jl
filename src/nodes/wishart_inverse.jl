import Distributions: InverseWishart

function make_node(::Type{ <: InverseWishart }; factorisation = ((1, 2, 3), ))
    return FactorNode(InverseWishart, Stochastic, (:out, :ν, :Ψ), factorisation, nothing)
end

function make_node(::Type{ <: InverseWishart }, out, ν, Ψ; factorisation = ((1, 2, 3), ))
    node = make_node(InverseWishart, factorisation = factorisation)
    connect!(node, :out, out)
    connect!(node, :ν, ν)
    connect!(node, :Ψ, Ψ)
    return node
end

