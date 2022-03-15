export AbstractModelSpecification
export ModelOptions, model_options
export FactorGraphModel
export AutoVar
export getoptions, getconstraints, getmeta
export getnodes, getrandom, getconstant, getdata
export activate!, repeat!
export UntilConvergence

import Base: show, getindex, haskey, firstindex, lastindex

# Abstract model specification

abstract type AbstractModelSpecification end

function create_model end

function model_name end

function source_code end

# Model Options

struct ModelOptions{P, F, S}
    pipeline                   :: P
    default_factorisation      :: F
    global_reactive_scheduler  :: S
end

model_options(; kwargs...)       = model_options(kwargs)
model_options(pairs::Base.Pairs) = model_options(NamedTuple(pairs))

available_option_names(::Type{ <: ModelOptions }) = (
    :pipeline, 
    :default_factorisation,
    :global_reactive_scheduler, 
    :limit_stack_depth
)

__as_named_tuple(nt::NamedTuple, arg1::NamedTuple{T, Tuple{Nothing}}, tail...) where T = __as_named_tuple(nt, tail...)
__as_named_tuple(nt::NamedTuple, arg1::NamedTuple, tail...)                            = __as_named_tuple(merge(nt, arg1), tail...)

__as_named_tuple(nt::NamedTuple) = nt

as_named_tuple(options::ModelOptions) = __as_named_tuple((;),
    (pipeline = options.pipeline, ),
    (default_factorisation = options.default_factorisation, ),
    (global_reactive_scheduler = options.global_reactive_scheduler, )
)

function model_options(options::NamedTuple)
    pipeline                  = nothing
    default_factorisation     = nothing
    global_reactive_scheduler = nothing

    if haskey(options, :pipeline)
        pipeline = options[:pipeline]
    end

    if haskey(options, :default_factorisation)
        default_factorisation = options[:default_factorisation]
    end

    if haskey(options, :global_reactive_scheduler)
        global_reactive_scheduler = options[:global_reactive_scheduler]
    elseif haskey(options, :limit_stack_depth)
        global_reactive_scheduler = LimitStackScheduler(options[:limit_stack_depth]...)
    end

    for key::Symbol in setdiff(union(available_option_names(ModelOptions), fields(options)), available_option_names(ModelOptions))
        @warn "Unknown option key: $key = $(options[key])"
    end

    return ModelOptions(
        pipeline,
        default_factorisation,
        global_reactive_scheduler
    )
end

global_reactive_scheduler(options::ModelOptions) = something(options.global_reactive_scheduler, AsapScheduler())
get_pipeline_stages(options::ModelOptions)       = something(options.pipeline, EmptyPipelineStage())
default_factorisation(options::ModelOptions)     = something(options.default_factorisation, UnspecifiedConstraints())

Base.merge(nt::NamedTuple, options::ModelOptions) = model_options(merge(nt, as_named_tuple(options)))

# Model

struct FactorGraphModel{C, M, O}
    constraints :: C
    meta        :: M
    options     :: O
    nodes       :: Vector{AbstractFactorNode}
    random      :: Vector{RandomVariable}
    constant    :: Vector{ConstVariable}
    data        :: Vector{DataVariable}
    vardict     :: Dict{Symbol, Any}
end

Base.show(io::IO, ::Type{ <: FactorGraphModel }) = print(io, "FactorGraphModel")
Base.show(io::IO, model::FactorGraphModel)       = print(io, "FactorGraphModel()")

FactorGraphModel() = FactorGraphModel(DefaultConstraints, DefaultMeta, model_options())

FactorGraphModel(constraints::Union{ UnspecifiedConstraints, ConstraintsSpecification }) = FactorGraphModel(constraints, DefaultMeta, model_options())
FactorGraphModel(meta::Union{ UnspecifiedMeta, MetaSpecification })                      = FactorGraphModel(DefaultConstraints, meta, model_options())
FactorGraphModel(options::NamedTuple)                                                    = FactorGraphModel(DefaultConstraints, DefaultMeta, model_options(options))

FactorGraphModel(constraints::Union{ UnspecifiedConstraints, ConstraintsSpecification }, options::NamedTuple) = FactorGraphModel(constraints, DefaultMeta, model_options(options))
FactorGraphModel(meta::Union{ UnspecifiedMeta, MetaSpecification }, options::NamedTuple)                      = FactorGraphModel(DefaultConstraints, meta, model_options(options))

FactorGraphModel(constraints::Union{ UnspecifiedConstraints, ConstraintsSpecification }, meta::Union{ UnspecifiedMeta, MetaSpecification })                       = FactorGraphModel(constraints, meta, model_options())
FactorGraphModel(constraints::Union{ UnspecifiedConstraints, ConstraintsSpecification }, meta::Union{ UnspecifiedMeta, MetaSpecification }, options::NamedTuple)  = FactorGraphModel(constraints, meta, model_options(options))

function FactorGraphModel(constraints::C, meta::M, options::O) where { C <: Union{ UnspecifiedConstraints, ConstraintsSpecification }, M <: Union{ UnspecifiedMeta, MetaSpecification }, O <: ModelOptions } 
    return FactorGraphModel{C, M, O}(constraints, meta, options, Vector{FactorNode}(), Vector{RandomVariable}(), Vector{ConstVariable}(), Vector{DataVariable}(), Dict{Symbol, Any}())
end

getconstraints(model::FactorGraphModel) = model.constraints
getmeta(model::FactorGraphModel)        = model.meta
getoptions(model::FactorGraphModel)     = model.options
getnodes(model::FactorGraphModel)       = model.nodes
getrandom(model::FactorGraphModel)      = model.random
getconstant(model::FactorGraphModel)    = model.constant
getdata(model::FactorGraphModel)        = model.data
getvardict(model::FactorGraphModel)     = model.vardict

function Base.getindex(model::FactorGraphModel, symbol::Symbol) 
    vardict = getvardict(model)
    if !haskey(vardict, symbol)
        error("Model has no variable/variables named $(symbol).")
    end
    return getindex(getvardict(model), symbol)
end

function Base.haskey(model::FactorGraphModel, symbol::Symbol)
    return haskey(getvardict(model), symbol)
end

firstindex(model::FactorGraphModel, symbol::Symbol) = firstindex(model, getindex(model, symbol))
lastindex(model::FactorGraphModel, symbol::Symbol)  = lastindex(model, getindex(model, symbol))

firstindex(::FactorGraphModel, ::AbstractVariable) = typemin(Int64)
lastindex(::FactorGraphModel, ::AbstractVariable)  = typemax(Int64)

firstindex(::FactorGraphModel, variables::AbstractVector{ <: AbstractVariable }) = firstindex(variables)
lastindex(::FactorGraphModel, variables::AbstractVector{ <: AbstractVariable })  = lastindex(variables)

add!(vardict::Dict, name::Symbol, entity) = vardict[name] = entity

add!(model::FactorGraphModel, node::AbstractFactorNode)  = begin push!(model.nodes, node); return node end
add!(model::FactorGraphModel, randomvar::RandomVariable) = begin push!(model.random, randomvar); add!(getvardict(model), name(randomvar), randomvar); return randomvar end
add!(model::FactorGraphModel, constvar::ConstVariable)   = begin push!(model.constant, constvar); add!(getvardict(model), name(constvar), constvar); return constvar end
add!(model::FactorGraphModel, datavar::DataVariable)     = begin push!(model.data, datavar); add!(getvardict(model), name(datavar), datavar); return datavar end
add!(model::FactorGraphModel, ::Nothing)                 = nothing
add!(model::FactorGraphModel, collection::Tuple)         = begin foreach((d) -> add!(model, d), collection); return collection end
add!(model::FactorGraphModel, array::AbstractArray)      = begin foreach((d) -> add!(model, d), array); return array end
add!(model::FactorGraphModel, array::AbstractArray{ <: RandomVariable }) = begin append!(model.random, array); add!(getvardict(model), name(first(array)), array); return array end
add!(model::FactorGraphModel, array::AbstractArray{ <: ConstVariable })  = begin append!(model.constant, array); add!(getvardict(model), name(first(array)), array); return array end
add!(model::FactorGraphModel, array::AbstractArray{ <: DataVariable })   = begin append!(model.data, array); add!(getvardict(model), name(first(array)), array); return array end

function activate!(model::FactorGraphModel) 
    filter!(getrandom(model)) do randomvar
        @assert degree(randomvar) !== 1 "Half-edge has been found: $(name(randomvar)). To terminate half-edges 'Uninformative' node can be used."
        return degree(randomvar) >= 2
    end

    foreach(getdata(model)) do datavar
        if !isconnected(datavar)
            @warn "Unused data variable has been found: '$(name(datavar))'. Ignore if '$(name(datavar))' has been used in deterministic nonlinear tranformation."
        end
    end

    filter!(c -> isconnected(c), getconstant(model))
    foreach(r -> activate!(model, r), getrandom(model))
    foreach(n -> activate!(model, n), getnodes(model))
end

# Utility functions

## node

function node_resolve_options(model::FactorGraphModel, options::FactorNodeCreationOptions, fform, variables) 
    return FactorNodeCreationOptions(
        node_resolve_factorisation(model, options, fform, variables),
        node_resolve_meta(model, options, fform, variables),
        getpipeline(options)
    )
end

## constraints 

node_resolve_factorisation(model::FactorGraphModel, options::FactorNodeCreationOptions, fform, variables)            = node_resolve_factorisation(model, options, factorisation(options), fform, variables)
node_resolve_factorisation(model::FactorGraphModel, options::FactorNodeCreationOptions, something, fform, variables) = something
node_resolve_factorisation(model::FactorGraphModel, options::FactorNodeCreationOptions, ::Nothing, fform, variables) = node_resolve_factorisation(model, getconstraints(model), default_factorisation(getoptions(model)), fform, variables)

node_resolve_factorisation(model::FactorGraphModel, constraints, default, fform, variables)                               = error("Cannot resolve factorisation constrains. Both `constraints` and `default_factorisation` option have been set, which is disallowed.")
node_resolve_factorisation(model::FactorGraphModel, ::UnspecifiedConstraints, default, fform, variables)                  = default
node_resolve_factorisation(model::FactorGraphModel, constraints, ::UnspecifiedConstraints, fform, variables)              = resolve_factorisation(constraints, model, fform, variables)
node_resolve_factorisation(model::FactorGraphModel, ::UnspecifiedConstraints, ::UnspecifiedConstraints, fform, variables) = resolve_factorisation(UnspecifiedConstraints(), model, fform, variables)

## meta 

node_resolve_meta(model::FactorGraphModel, options::FactorNodeCreationOptions, fform, variables)            = node_resolve_meta(model, options, metadata(options), fform, variables)
node_resolve_meta(model::FactorGraphModel, options::FactorNodeCreationOptions, something, fform, variables) = something
node_resolve_meta(model::FactorGraphModel, options::FactorNodeCreationOptions, ::Nothing, fform, variables) = resolve_meta(getmeta(model), model, fform, variables)

## randomvar

function randomvar_resolve_options(model::FactorGraphModel, options::RandomVariableCreationOptions, name) 
    qform, qprod = randomvar_resolve_marginal_form_prod(model, options, name)
    mform, mprod = randomvar_resolve_messages_form_prod(model, options, name)

    rprod = resolve_prod_constraint(qprod, mprod)

    qoptions = randomvar_options_set_marginal_form_constraint(options, qform)
    moptions = randomvar_options_set_messages_form_constraint(qoptions, mform)
    roptions = randomvar_options_set_prod_constraint(moptions, rprod)

    return roptions
end

## constraints

randomvar_resolve_marginal_form_prod(model::FactorGraphModel, options::RandomVariableCreationOptions, name)            = randomvar_resolve_marginal_form_prod(model, options, marginal_form_constraint(options), name)
randomvar_resolve_marginal_form_prod(model::FactorGraphModel, options::RandomVariableCreationOptions, something, name) = (something, nothing)
randomvar_resolve_marginal_form_prod(model::FactorGraphModel, options::RandomVariableCreationOptions, ::Nothing, name) = randomvar_resolve_marginal_form_prod(model, getconstraints(model), name)

randomvar_resolve_marginal_form_prod(model::FactorGraphModel, ::UnspecifiedConstraints, name) = (nothing, nothing)
randomvar_resolve_marginal_form_prod(model::FactorGraphModel, constraints, name)              = resolve_marginal_form_prod(constraints, model, name)

randomvar_resolve_messages_form_prod(model::FactorGraphModel, options::RandomVariableCreationOptions, name)            = randomvar_resolve_messages_form_prod(model, options, messages_form_constraint(options), name)
randomvar_resolve_messages_form_prod(model::FactorGraphModel, options::RandomVariableCreationOptions, something, name) = (something, nothing)
randomvar_resolve_messages_form_prod(model::FactorGraphModel, options::RandomVariableCreationOptions, ::Nothing, name) = randomvar_resolve_messages_form_prod(model, getconstraints(model), name)

randomvar_resolve_messages_form_prod(model::FactorGraphModel, ::UnspecifiedConstraints, name) = (nothing, nothing)
randomvar_resolve_messages_form_prod(model::FactorGraphModel, constraints, name)              = resolve_messages_form_prod(constraints, model, name)

## variable creation

function randomvar(model::FactorGraphModel, name::Symbol, args...)
    return randomvar(model, RandomVariableCreationOptions(), name, args...)
end

function randomvar(model::FactorGraphModel, options::RandomVariableCreationOptions, name::Symbol, args...) 
    return add!(model, randomvar(randomvar_resolve_options(model, options, name), name, args...))
end

constvar(model::FactorGraphModel, args...)  = add!(model, constvar(args...))
datavar(model::FactorGraphModel, args...)   = add!(model, datavar(args...))

as_variable(model::FactorGraphModel, x)                   = add!(model, as_variable(x))
as_variable(model::FactorGraphModel, v::AbstractVariable) = v
as_variable(model::FactorGraphModel, t::Tuple)            = map((d) -> as_variable(model, d), t)

function make_node(model::FactorGraphModel, options::FactorNodeCreationOptions, fform, args...)
    return add!(model, make_node(fform, node_resolve_options(model, options, fform, args), args...))
end

## AutoVar 

struct AutoVar
    name :: Symbol
end

getname(autovar::AutoVar) = autovar.name

function ReactiveMP.make_node(model::FactorGraphModel, options::FactorNodeCreationOptions, fform, autovar::AutoVar, args::Vararg{ <: ReactiveMP.AbstractVariable })
    proxy     = isdeterministic(sdtype(fform)) ? args : nothing
    rvoptions = ReactiveMP.randomvar_options_set_proxy_variables(EmptyRandomVariableCreationOptions, proxy)
    var       = ReactiveMP.randomvar(model, rvoptions, ReactiveMP.getname(autovar)) # add! is inside
    node      = ReactiveMP.make_node(model, options, fform, var, args...) # add! is inside
    return node, var
end

__fform_const_apply(::Type{T}, args...) where T = T(args...)
__fform_const_apply(f::F, args...) where { F <: Function } = f(args...)

function ReactiveMP.make_node(model::FactorGraphModel, options::FactorNodeCreationOptions, fform, autovar::AutoVar, args::Vararg{ <: ReactiveMP.ConstVariable })
    if isstochastic(sdtype(fform))
        var  = ReactiveMP.randomvar(model, EmptyRandomVariableCreationOptions, ReactiveMP.getname(autovar)) # add! is inside
        node = ReactiveMP.make_node(model, options, fform, var, args...) # add! is inside
        return node, var
    else
        var  = add!(model, ReactiveMP.constvar(ReactiveMP.getname(autovar), __fform_const_apply(fform, map((d) -> ReactiveMP.getconst(d), args)...)))
        return nothing, var
    end
end

##

# Repeat variational message passing iterations [ EXPERIMENTAL ]

repeat!(model::FactorGraphModel, criterion) = repeat!((_) -> nothing, model, criterion)

function repeat!(callback::Function, model::FactorGraphModel, count::Int)
    for _ in 1:count
        foreach(getdata(model)) do datavar
            resend!(datavar)
        end
        callback(model)
    end
end

struct UntilConvergence{S, F}
    score_type :: S
    tolerance  :: F
    maxcount   :: Int
    mincount   :: Int
end

UntilConvergence(score::S; tolerance::F = 1e-6, maxcount::Int = 10_000, mincount::Int = 0) where { S, F <: Real } = UntilConvergence{S, F}(score, tolerance, maxcount, mincount)
UntilConvergence(; kwargs...)                                                                                     = UntilConvergence(BetheFreeEnergy(); kwargs...)

function repeat!(callback::Function, model::FactorGraphModel, criterion::UntilConvergence{S, F}) where { S, F }

    stopping_fn = let tolerance = criterion.tolerance
        (p) -> abs(p[1] - p[2]) < tolerance
    end

    source = score(F, criterion.score_type, model, AsapScheduler()) |> pairwise() |> map(Bool, stopping_fn)

    is_satisfied     = false
    iterations_count = 0

    subscription = subscribe!(source, lambda(
        on_next     = (v) -> is_satisfied = v,
        on_complete = (v) -> is_satisfied = true
    ))

    while !is_satisfied
        foreach(getdata(model)) do datavar
            resend!(datavar)
        end
        callback(model)
        iterations_count += 1

        if iterations_count <= criterion.mincount
            is_satisfied = false
        end

        if iterations_count >= criterion.maxcount
            is_satisfied = true
        end
    end

    unsubscribe!(subscription)

    return iterations_count
end