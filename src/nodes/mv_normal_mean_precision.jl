export make_node, rule

function make_node(::Type{ <: MvNormalMeanPrecision }; factorisation = ((1, 2, 3), ))
    return FactorNode(MvNormalMeanPrecision, Stochastic, (:out, :μ, :Λ), factorisation, nothing)
end

function make_node(::Type{ <: MvNormalMeanPrecision }, out, μ, Λ; factorisation = ((1, 2, 3), ))
    node = make_node(MvNormalMeanPrecision, factorisation = factorisation)
    connect!(node, :out, out)
    connect!(node, :μ, μ)
    connect!(node, :Λ, Λ)
    return node
end