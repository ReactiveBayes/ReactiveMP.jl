export DataVariable, datavar, update!

import Base: show

struct DataVariableProperties
    input_messages::Vector{MessageObservable{AbstractMessage}}
    messageout
    prediction
end

function DataVariableProperties()
    return DataVariableProperties(Vector{MessageObservable{AbstractMessage}}(), RecentSubject(Message), MarginalObservable())
end

## Old stuff is below

mutable struct DataVariable{D, S} <: AbstractVariable
    name            :: Symbol
    collection_type :: AbstractVariableCollectionType
    prediction      :: MarginalObservable
    input_messages  :: Vector{MessageObservable{AbstractMessage}}
    messageout      :: S
    nconnected      :: Int
    isproxy         :: Bool
    isused          :: Bool
end

Base.show(io::IO, datavar::DataVariable) = print(io, "DataVariable(", indexed_name(datavar), ")")

struct DataVariableCreationOptions{S}
    subject :: S
    isproxy :: Bool
    isused  :: Bool
end

Base.similar(options::DataVariableCreationOptions) = DataVariableCreationOptions(similar(options.subject), options.isproxy, options.isused)

DataVariableCreationOptions(::Type{D}) where {D}          = DataVariableCreationOptions(D, nothing)
DataVariableCreationOptions(::Type{D}, subject) where {D} = DataVariableCreationOptions(D, subject, Val(false))

DataVariableCreationOptions(::Type{D}, subject::Nothing, allow_missing::Val{true}) where {D}  = DataVariableCreationOptions(D, RecentSubject(Union{Message{Missing}, Message{D}}), Val(false)) # `Val(false)` here is intentional
DataVariableCreationOptions(::Type{D}, subject::Nothing, allow_missing::Val{false}) where {D} = DataVariableCreationOptions(D, RecentSubject(Union{Message{D}}), Val(false))

DataVariableCreationOptions(::Type{D}, subject::S, ::Val{true}) where {D, S}  = error("Error in datavar options. Custom `subject` was specified and `allow_missing` was set to true, which is disallowed. Provide a custom subject that accept missing values by itself and do no use `allow_missing` option.")
DataVariableCreationOptions(::Type{D}, subject::S, ::Val{false}) where {D, S} = DataVariableCreationOptions{S}(subject, false, false)

""" 
    datavar(::Type, [ dims... ])

`datavar()` function provides a mechanism to pass data values to the model. You can create data inputs with `datavar()` function. 
As a first argument it accepts a type specification and optional dimensionality (as additional arguments or as a tuple). 
User can treat `datavar()`s in the model as both clamped values for priors and observations.

Note: `datavar()` function is supposed to be used only within the `@model` macro, see `GraphPPL.jl` package.

## Example 

```julia
@model function model_name(...)
    ...
    y = datavar(Float64) # Creates a single data input with `y` as identificator
    y = datavar(Float64, n) # Returns a vector of  `y_i` data input objects with length `n`
    y = datavar(Float64, n, m) # Returns a matrix of `y_i_j` data input objects with size `(n, m)`
    y = datavar(Float64, (n, m)) # It is also possible to use a tuple for dimensionality
    ...
end
```

`datavar()` call within `@model` macro supports `where { options... }` block for extra options specification, e.g:

```julia
@model function model_name(...)
    ...
    y = datavar(Float64, n) where { allow_missing = true }
    ...
end
```

### Data variables available options

- `allow_missing = true/false`: Specifies if it is possible to pass `missing` object as an observation. Note, however, that by default ReactiveMP.jl does not expose any message computation rules that involve `missing`s.
"""
function datavar end

datavar(name::Symbol, ::Type{D}, collection_type::AbstractVariableCollectionType = VariableIndividual()) where {D} = datavar(DataVariableCreationOptions(D), name, D, collection_type)
datavar(name::Symbol, ::Type{D}, length::Int) where {D}                                                            = datavar(DataVariableCreationOptions(D), name, D, length)
datavar(name::Symbol, ::Type{D}, dims::Tuple) where {D}                                                            = datavar(DataVariableCreationOptions(D), name, D, dims)
datavar(name::Symbol, ::Type{D}, dims::Vararg{Int}) where {D}                                                      = datavar(DataVariableCreationOptions(D), name, D, dims)

datavar(options::DataVariableCreationOptions{S}, name::Symbol, ::Type{D}, collection_type::AbstractVariableCollectionType = VariableIndividual()) where {S, D} =
    DataVariable{D, S}(name, collection_type, MarginalObservable(), Vector{MessageObservable{AbstractMessage}}(), options.subject, 0, options.isproxy, options.isused)

function datavar(options::DataVariableCreationOptions, name::Symbol, ::Type{D}, length::Int) where {D}
    return map(i -> datavar(similar(options), name, D, VariableVector(i)), 1:length)
end

function datavar(options::DataVariableCreationOptions, name::Symbol, ::Type{D}, dim1::Int, extra_dims::Vararg{Int}) where {D}
    return datavar(options, name, D, (dim1, extra_dims...))
end

function datavar(options::DataVariableCreationOptions, name::Symbol, ::Type{D}, dims::Tuple) where {D}
    indices = CartesianIndices(dims)
    size    = axes(indices)
    return map(i -> datavar(similar(options), name, D, VariableArray(size, i)), indices)
end

Base.eltype(::Type{<:DataVariable{D}}) where {D} = D
Base.eltype(::DataVariable{D}) where {D}         = D

degree(datavar::DataVariable)          = nconnected(datavar)
name(datavar::DataVariable)            = datavar.name
proxy_variables(datavar::DataVariable) = nothing    # not related to isproxy
collection_type(datavar::DataVariable) = datavar.collection_type
isconnected(datavar::DataVariable)     = datavar.nconnected !== 0
nconnected(datavar::DataVariable)      = datavar.nconnected
setused!(datavar::DataVariable)        = datavar.isused = true

isproxy(datavar::DataVariable) = datavar.isproxy
isused(datavar::DataVariable) = datavar.isused

israndom(::DataVariable)                  = false
israndom(::AbstractArray{<:DataVariable}) = false
isdata(::DataVariable)                    = true
isdata(::AbstractArray{<:DataVariable})   = true
isconst(::DataVariable)                   = false
isconst(::AbstractArray{<:DataVariable})  = false

allows_missings(datavar::DataVariable) = allows_missings(datavar, eltype(datavar.messageout))

allows_missings(datavars::AbstractArray{<:DataVariable}) = all(allows_missings, datavars)
allows_missings(datavar::DataVariable, ::Type{Message{D}}) where {D} = false
allows_missings(datavar::DataVariable, ::Type{Union{Message{Missing}, Message{D}}} where {D}) = true

function Base.getindex(datavar::DataVariable, i...)
    error("Variable $(indexed_name(datavar)) has been indexed with `[$(join(i, ','))]`. Direct indexing of `data` variables is not allowed.")
end

getlastindex(datavar::DataVariable) = degree(datavar) + 1

messageout(datavar::DataVariable, ::Int) = datavar.messageout
messagein(datavar::DataVariable, ::Int)  = error("It is not possible to get a reference for inbound message for datavar")

update!(datavar::DataVariable, ::Missing)           = next!(messageout(datavar, 1), Message(missing, false, false, nothing))
update!(datavar::DataVariable, data::Number)        = update!(eltype(datavar), datavar, data)
update!(datavar::DataVariable, data::AbstractArray) = update!(eltype(datavar), datavar, data)

update!(::Type{D}, datavar, data::D) where {D}        = next!(messageout(datavar, 1), Message(data, false, false, nothing))
update!(::Type{D1}, datavar, data::D2) where {D1, D2} = __update_wrong_type_error(D1, D2, collection_type(datavar), datavar)

__datavar_drop_pointmass(::Type{D}) where {D}            = D
__datavar_drop_pointmass(::Type{PointMass{D}}) where {D} = D

__update_wrong_type_error(::Type{D1}, ::Type{D2}, ctype::VariableIndividual, datavar) where {D1, D2} = error(
    """
    `$(name(datavar)) = datavar($(__datavar_drop_pointmass(D1)))` accepts data only of type `$(__datavar_drop_pointmass(D1))`, but the value of type `$D2` has been used. 
    Double check `update!($(name(datavar)), data)` call and explicitly convert data to the type `$(__datavar_drop_pointmass(D1))`, e.g. `convert($(__datavar_drop_pointmass(D1)), data)`.
    """
)

__update_wrong_type_error(::Type{D1}, ::Type{D2}, ctype::Union{VariableVector, VariableArray}, datavar) where {D1, D2} = error(
    """
    `$(name(datavar)) = datavar($(__datavar_drop_pointmass(D1)), ...)` accepts data only of type `$(__datavar_drop_pointmass(D1))`, but the value of type `$D2` has been used. 
    Double check `update!($(name(datavar))$(string_index(ctype)), d)` call and explicitly convert data to the type `$(__datavar_drop_pointmass(D1))`, e.g. `update!($(name(datavar))$(string_index(ctype)), convert($(__datavar_drop_pointmass(D1)), d))`.
    If you use broadcasted version of the `update!` function, e.g. `update!($(name(datavar)), data)` you may broadcast `convert` function over the whole dataset as well, e.g. `update!($(name(datavar)), convert.($(__datavar_drop_pointmass(D1)), dataset))`
    """
)

update!(::Type{PointMass{D}}, datavar, data::D) where {D} = next!(messageout(datavar, 1), Message(PointMass(data), false, false, nothing))

resend!(datavar::DataVariable) = update!(datavar, Rocket.getrecent(messageout(datavar, 1)))

function update!(datavars::AbstractArray{<:DataVariable}, data::AbstractArray)
    @assert size(datavars) === size(data) """
    Invalid `update!` call: size of datavar array and data must match: `$(name(first(datavars)))` has size $(size(datavars)) and data has size $(size(data)). 
    """
    foreach(zip(datavars, data)) do (var, d)
        update!(var, d)
    end
end

finish!(datavar::DataVariable) = complete!(messageout(datavar, 1))

get_pipeline_stages(::DataVariable) = EmptyPipelineStage()

_getmarginal(datavar::DataVariable)              = datavar.messageout |> map(Marginal, as_marginal)
_setmarginal!(datavar::DataVariable, observable) = error("It is not possible to set a marginal stream for `DataVariable`")
_makemarginal(datavar::DataVariable)             = error("It is not possible to make marginal stream for `DataVariable`")

setanonymous!(::DataVariable, ::Bool) = nothing

function setmessagein!(datavar::DataVariable, index::Int, messagein)
    if index === (degree(datavar) + 1)
        push!(datavar.input_messages, messagein)
        datavar.nconnected += 1
        datavar.isused = true
    else
        error(
            "Inconsistent state in setmessagein! function for data variable $(datavar). `index` should be equal to `degree(datavar) + 1 = $(degree(datavar) + 1)`, $(index) is given instead"
        )
    end
    return nothing
end

marginal_prod_fn(datavar::DataVariable) = marginal_prod_fn(FoldLeftProdStrategy(), GenericProd(), UnspecifiedFormConstraint(), FormConstraintCheckLast())

_getprediction(datavar::DataVariable)              = datavar.prediction
_setprediction!(datavar::DataVariable, observable) = connect!(_getprediction(datavar), observable)
_makeprediction(datavar::DataVariable)             = collectLatest(AbstractMessage, Marginal, datavar.input_messages, marginal_prod_fn(datavar))

function activate!(datavar::DataVariable, options)
    _setprediction!(datavar, _makeprediction(datavar))

    return nothing
end
