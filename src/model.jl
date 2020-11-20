export ModelOptions, model_options
export Model
export getnodes, getrandom, getconstant, getdata
export activate!
# export MarginalsEagerUpdate, MarginalsPureUpdate

import Base: show

# Marginals update strategies 

# struct MarginalsEagerUpdate end
# struct MarginalsPureUpdate end

# Model Options

struct ModelOptions{P, F}
    outbound_message_portal :: P
    default_factorisation   :: F
end

model_options() = model_options(NamedTuple{()}(()))

function model_options(options::NamedTuple)
    outbound_message_portal = DefaultOutboundMessagePortal()
    default_factorisation   = FullFactorisation()

    if haskey(options, :outbound_message_portal)
        outbound_message_portal = options[:outbound_message_portal]
    end

    if haskey(options, :default_factorisation)
        default_factorisation = options[:default_factorisation]
    end

    for key::Symbol in setdiff(union(fieldnames(ModelOptions), fields(options)), fieldnames(ModelOptions))
        @warn "Unknown option key: $key = $(options[key])"
    end

    return ModelOptions(
        outbound_message_portal,
        default_factorisation
    )
end

outbound_message_portal(options::ModelOptions) = options.outbound_message_portal
default_factorisation(options::ModelOptions)   = options.default_factorisation

# Model

struct Model{O}
    options  :: O
    nodes    :: Vector{FactorNode}
    random   :: Vector{RandomVariable}
    constant :: Dict{Symbol, ConstVariable}
    data     :: Dict{Symbol, DataVariable}
end

Base.show(io::IO, model::Model) = print(io, "Model($(getoptions(model)))")

Model() = Model(model_options())
Model(options::NamedTuple) = Model(model_options(options))
Model(options::O) where { O <: ModelOptions } = Model{O}(options, Vector{FactorNode}(), Vector{RandomVariable}(), Dict{Symbol, ConstVariable}(), Dict{Symbol, DataVariable}())

getoptions(model::Model)  = model.options
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
            @warn "Unused data variable has been found: '$(name(datavar))'. Ignore if '$(name(datavar))' has been used in deterministic nonlinear tranformation."
        end
    end

    filter!(c -> isconnected(last(c)), getconstant(model))
    foreach(n -> activate!(model, n), getnodes(model))
end

# Utility functions

randomvar(model::Model, args...; kwargs...) = add!(model, randomvar(args...; kwargs...))

function make_node(model::Model, args...; factorisation = default_factorisation(getoptions(model)), kwargs...) 
    return add!(model, make_node(args...; factorisation = factorisation, kwargs...))
end

function make_node(model::Model, fform, autovar::AutoVar, args::Vararg{ <: ConstVariable{ <: Dirac } }; factorisation = default_factorisation(getoptions(model)), kwargs...)
    node, var = if haskey(getconstant(model), getname(autovar))
        nothing, getconstant(model)[ getname(autovar) ]
    else
        add!(model, make_node(fform, autovar, args...; factorisation = factorisation, kwargs...))
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