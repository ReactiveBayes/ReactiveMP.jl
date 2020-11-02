export Model, add!
export getnodes, getrandom, getconstant, getdata
export activate!
# export MarginalsEagerUpdate, MarginalsPureUpdate

import Base: show

# Marginals update strategies 

# struct MarginalsEagerUpdate end
# struct MarginalsPureUpdate end

struct Model{T, S}
    message_gate :: T
    message_out_transformer :: S

    nodes    :: Vector{FactorNode}
    random   :: Vector{RandomVariable}
    constant :: Vector{ConstVariable}
    data     :: Vector{DataVariable}
end

Base.show(io::IO, ::Model) = print(io, "Model()")

Model(; 
    message_gate::T            = DefaultMessageGate(),
    message_out_transformer::S = DefaultMessageOutTransformer()
) where { T, S } = Model{T, S}(
    message_gate, 
    message_out_transformer,
    Vector{FactorNode}(), 
    Vector{RandomVariable}(),
    Vector{ConstVariable}(),
    Vector{DataVariable}()
)

message_gate(model::Model)            = model.message_gate
message_out_transformer(model::Model) = model.message_out_transformer

# placeholder for future
add!(model::Model, node::FactorNode)        = begin push!(model.nodes, node); return node end
add!(model::Model, random::RandomVariable)  = begin push!(model.random, random); return random end
add!(model::Model, constant::ConstVariable) = begin push!(model.constant, constant); return constant end
add!(model::Model, data::DataVariable)      = begin push!(model.data, data); return data end
add!(model::Model, ::Nothing)               = begin return nothing end
add!(model::Model, collection::Tuple)       = begin foreach((d) -> add!(model, d), collection); return collection end
add!(model::Model, array::AbstractArray)    = begin foreach((d) -> add!(model, d), array); return array end

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