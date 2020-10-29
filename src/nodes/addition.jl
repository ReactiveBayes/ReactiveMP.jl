export make_node, rule

using Distributions

function make_node(::typeof(+)) 
    return FactorNode(+, Deterministic, (:out, :in1, :in2), ((1, 2, 3), ), nothing)
end

function make_node(::typeof(+), out::AbstractVariable, in1::AbstractVariable, in2::AbstractVariable)
    node = make_node(+)
    connect!(node, :out, out)
    connect!(node, :in1, in1)
    connect!(node, :in2, in2)
    return node
end
