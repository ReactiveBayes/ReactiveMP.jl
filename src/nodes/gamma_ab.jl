export make_node, rule

function GammaABNode(::Type{T} = Float64; factorisation = ((1, 2, 3), )) where T
    return FactorNode(GammaAB{T}, Stochastic, (:a, :b, :value), factorisation)
end

function make_node(::Type{ <: GammaAB{T} }; factorisation = ((1, 2, 3), )) where T
    return GammaABNode(T, factorisation = factorisation)
end

function make_node(::Type{ <: GammaAB{T} }, a, b, value; factorisation = ((1, 2, 3), )) where T
    node = make_node(GammaAB{T}, factorisation = factorisation)
    connect!(node, :a, a)
    connect!(node, :b, b)
    connect!(node, :value, value)
    return node
end

## rules

function rule(::Type{ <: GammaAB{T} }, ::Type{ Val{:value} }, ::Marginalisation, messages::Tuple{Message{T}, Message{T}}, marginals::Nothing, meta) where { T <: Real }
    return GammaAB{T}(mean(messages[1]), mean(messages[2]))
end
