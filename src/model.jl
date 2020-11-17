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
    constant :: Dict{Symbol, ConstVariable}
    data     :: Dict{Symbol, DataVariable}
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
    Dict{Symbol, ConstVariable}(),
    Dict{Symbol, DataVariable}()
)

message_gate(model::Model)            = model.message_gate
message_out_transformer(model::Model) = model.message_out_transformer

getnodes(model::Model)    = model.nodes
getrandom(model::Model)   = model.random
getconstant(model::Model) = model.constant
getdata(model::Model)     = model.data

# placeholder for future
add!(model::Model, node::FactorNode)        = begin push!(model.nodes, node); return node end
add!(model::Model, random::RandomVariable)  = begin push!(model.random, random); return random end
add!(model::Model, ::Nothing)               = nothing
add!(model::Model, collection::Tuple)       = begin foreach((d) -> add!(model, d), collection); return collection end
add!(model::Model, array::AbstractArray)    = begin foreach((d) -> add!(model, d), array); return array end

function add!(model::Model, constant::ConstVariable)
    @assert !haskey(getconstant(model), name(constant)) "ConstVariable with name '$(name(constant))' has already been added to a model"
    model.constant[ name(constant) ] = constant
    return constant
end

function add!(model::Model, data::DataVariable)
    @assert !haskey(getdata(model), name(data)) "DataVariable with name '$(name(data))' has already been added to a model"
    model.data[ name(data) ] = data
    return data
end

function activate!(model::Model) 
    filter!(getrandom(model)) do randomvar
        @assert degree(randomvar) !== 1 "Loose random variable has been found: $(name(randomvar))"
        return degree(randomvar) >= 2
    end

    foreach(values(getdata(model))) do datavar
        if !isconnected(datavar)
            @warn "Unused data variable has been found: $(name(datavar))"
        end
    end

    filter!(c -> isconnected(last(c)), getconstant(model))
    foreach(n -> activate!(model, n), getnodes(model))
end

# Utility functions

randomvar(model::Model, args...; kwargs...) = add!(model, randomvar(args...; kwargs...))
make_node(model::Model, args...; kwargs...) = add!(model, make_node(args...; kwargs...))

function make_node(model::Model, fform, autovar::AutoVar, args::Vararg{ <: ConstVariable{ <: Dirac } }; kwargs...)
    node, var = if haskey(getconstant(model), getname(autovar))
        nothing, getconstant(model)[ getname(autovar) ]
    else
        add!(model, make_node(fform, autovar, args...; kwargs...))
    end
end

constvar(model::Model, name::Symbol, constval)  = get!(() -> constvar(name, constval), getconstant(model), name)
datavar(model::Model, name::Symbol, type::Type) = get!(() -> datavar(name, type), getdata(model), name)

function datavar(model::Model, nameprefix::Symbol, type::Type, dims...) 
    datavars = datavar(nameprefix, type, dims...)
    dictvars = Dict(zip(map(name, datavars), datavars))
    mergewith!((l, r) -> error("DataVariable with name $(name(l)) has already been added to the model"), getdata(model), dictvars)
    return datavars
end