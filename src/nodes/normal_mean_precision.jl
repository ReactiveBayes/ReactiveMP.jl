export make_node, rule


function make_node(::Type{ <: NormalMeanPrecision }; factorisation = ((1, 2, 3), ))
    return FactorNode(NormalMeanPrecision, Stochastic, (:out, :μ, :τ), factorisation, nothing)
end

function make_node(::Type{ <: NormalMeanPrecision }, out, μ, τ; factorisation = ((1, 2, 3), ))
    node = make_node(NormalMeanPrecision, factorisation = factorisation)
    connect!(node, :out, out)
    connect!(node, :μ, μ)
    connect!(node, :τ, τ)
    return node
end
