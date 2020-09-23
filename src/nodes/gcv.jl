export make_node, rule, GCV

struct GCV end

function make_node(::Type{ GCV })
    return FactorNode(GCV, Stochastic, ( :y, :x, :z, :κ, :ω ), ( ( 1, 2 ), ( 3, ), ( 4, ), ( 5, ) ), nothing)
end

function make_node(::Type{ GCV }, y, x, z, κ, ω)
    node = make_node(GCV)
    connect!(node, :y, y)
    connect!(node, :x, x)
    connect!(node, :z, z)
    connect!(node, :κ, κ)
    connect!(node, :ω, ω)
    return node
end

