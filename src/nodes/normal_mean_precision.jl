export make_node, rule


function make_node(::Type{ <: NormalMeanPrecision }; factorisation = ((1, 2, 3), ))
    return FactorNode(NormalMeanPrecision, Stochastic, (:out, :mean, :precision), factorisation, nothing)
end

function make_node(::Type{ <: NormalMeanPrecision }, out, mean, precision; factorisation = ((1, 2, 3), ))
    node = make_node(NormalMeanPrecision, factorisation = factorisation)
    connect!(node, :out, out)
    connect!(node, :mean, mean)
    connect!(node, :precision, precision)
    return node
end
