export Model, add!

import Base: show

struct Model{T}
    message_gate :: T

    nodes    :: Vector{FactorNode}
    random   :: Vector{RandomVariable}
    constant :: Vector{ConstVariable}
    data     :: Vector{DataVariable}
end

Base.show(io::IO, model::Model) = print(io, "Model()")

Model(message_gate::T) where T = Model{T}(
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
