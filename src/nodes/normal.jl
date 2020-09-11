export make_node, rule

import Distributions: Normal

function make_node(::Type{ <: Normal })
    return FactorNode(Normal, Stochastic, (:out, :mean, :std), ((1, 2, 3), ), nothing)
end

function make_node(::Type{ <: Normal }, out, mean, std)
    node = make_node(Normal)
    connect!(node, :out, out)
    connect!(node, :mean, mean)
    connect!(node, :std, std)
    return node
end
