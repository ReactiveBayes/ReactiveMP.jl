export make_node, rule


function make_node(::Type{ <: GammaAB }; factorisation = ((1, 2, 3), ))
    return FactorNode(GammaAB, Stochastic, (:out, :a, :b), factorisation, nothing)
end

function make_node(::Type{ <: GammaAB }, out, a, b; factorisation = ((1, 2, 3), ))
    node = make_node(GammaAB, factorisation = factorisation)
    connect!(node, :out, out)
    connect!(node, :a, a)
    connect!(node, :b, b)
    return node
end

## rules

function rule(::Type{ <: GammaAB }, ::Type{ Val{:out} }, ::Marginalisation, messages::Tuple{Message{T}, Message{T}}, ::Nothing, ::Nothing) where { T <: Real }
    return GammaAB(mean(messages[1]), mean(messages[2]))
end

## marginalrules 

function marginalrule(::Type{ <: GammaAB }, ::Type{ Val{ :out_a_b } }, messages::Tuple{Message{GammaAB{T}}, Message{T}, Message{T}}, ::Nothing, ::Nothing) where { T <: Real }
    q_out = Message(GammaAB(mean(messages[2]), mean(messages[3]))) * messages[1]
    return (getdata(q_out), getdata(messages[2]), getdata(messages[3]))
end
