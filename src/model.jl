export Model, add!
export getnodes, getrandom, getconstant, getdata
export activate!
# export MarginalsEagerUpdate, MarginalsPureUpdate

import Base: show

# Marginals update strategies 

# struct MarginalsEagerUpdate end
# struct MarginalsPureUpdate end

struct Model{T}
    message_gate :: T

    nodes    :: Vector{FactorNode}
    random   :: Vector{RandomVariable}
    constant :: Vector{ConstVariable}
    data     :: Vector{DataVariable}
end

Base.show(io::IO, ::Model) = print(io, "Model()")

Model(; 
    message_gate::T = DefaultMessageGate()
) where { T } = Model{T}(
    message_gate, 
    Vector{FactorNode}(), 
    Vector{RandomVariable}(),
    Vector{ConstVariable}(),
    Vector{DataVariable}()
)

message_gate(model::Model) = model.message_gate

# placeholder for future
add!(model, node::FactorNode)        = begin push!(model.nodes, node); return node end
add!(model, random::RandomVariable)  = begin push!(model.random, random); return random end
add!(model, constant::ConstVariable) = begin push!(model.constant, constant); return constant end
add!(model, data::DataVariable)      = begin push!(model.data, data); return data end

getnodes(model::Model)    = model.nodes
getrandom(model::Model)   = model.random
getconstant(model::Model) = model.constant
getdata(model::Model)     = model.data

activate!(model::Model) = foreach(n -> activate!(model, n), getnodes(model))

# Utility functions

datavar(model::Model, args...; kwargs...)   = add!(model, datavar(args...; kwargs...))
constvar(model::Model, args...; kwargs...)  = add!(model, constvar(args...; kwargs...))
randomvar(model::Model, args...; kwargs...) = add!(model, randomvar(args...; kwargs...))
make_node(model::Model, args...; kwargs...) = add!(model, make_node(args...; kwargs...))
