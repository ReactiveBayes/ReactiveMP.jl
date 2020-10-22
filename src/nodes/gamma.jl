export make_node, rule


function make_node(::Type{ <: Gamma }; factorisation = ((1, 2, 3), ))
    return FactorNode(Gamma, Stochastic, (:out, :α, :θ), factorisation, nothing)
end

function make_node(::Type{ <: Gamma }, out, α, θ; factorisation = ((1, 2, 3), ))
    node = make_node(Gamma, factorisation = factorisation)
    connect!(node, :out, out)
    connect!(node, :α, α)
    connect!(node, :θ, θ)
    return node
end

