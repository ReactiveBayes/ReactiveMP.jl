export make_node, rule

function GammaABNode(::Type{T} = Float64; factorisation = ((1, 2, 3), )) where T
    return FactorNode(GammaAB{T}, Stochastic, (:a, :b, :out), factorisation)
end

function make_node(::Type{ <: GammaAB{T} }; factorisation = ((1, 2, 3), )) where T
    return GammaABNode(T, factorisation = factorisation)
end

function make_node(::Type{ <: GammaAB{T} }, a, b, out; factorisation = ((1, 2, 3), )) where T
    node = make_node(GammaAB{T}, factorisation = factorisation)
    connect!(node, :a, a)
    connect!(node, :b, b)
    connect!(node, :out, out)
    return node
end

## rules

function rule(::Type{ <: GammaAB{T} }, ::Type{ Val{:out} }, ::Marginalisation, messages::Tuple{Message{T}, Message{T}}, marginals::Nothing, meta) where { T <: Real }
    return GammaAB{T}(mean(messages[1]), mean(messages[2]))
end

## marginalrules 

function marginalrule(::Type{ <: GammaAB{T} }, ::Type{ Val{ :a_b_out } }, messages::Tuple{Message{T}, Message{T}, Message{GammaAB{T}}}, ::Nothing, ::Nothing) where { T <: Real }
    q_out = Message(GammaAB(getdata(messages[1]), getdata(messages[2]))) * messages[3]
    return (getdata(messages[1]), getdata(messages[2]), getdata(q_out))
end
