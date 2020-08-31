export make_node, rule

import Distributions: Normal

# BP rule

# TODO
make_node(::Type{ <: Normal{T} }) where T = FactorNode(Normal{T}, Stochastic, (:mean, :std, :value), ((1, 2, 3), ))

function make_node(::Type{ <: Normal{T} }, mean, std, value) where T 
    node = make_node(Normal{T})
    connect!(node, :mean, mean)
    connect!(node, :std, std)
    connect!(node, :value, value)
    return node
end

function rule(::Type{ <: Normal{T} }, ::Type{ Val{:value} }, ::Marginalisation, messages::Tuple{Message{T}, Message{T}}, marginals::Nothing, meta) where { T <: Real }
    return Normal{T}(mean(messages[1]), sqrt(mean(messages[2])))
end
