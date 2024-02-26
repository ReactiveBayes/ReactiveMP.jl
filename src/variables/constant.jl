export ConstVariable, constvar, getconst, isconnected

import Rocket: SingleObservable, AsapScheduler
import Base: getindex, show

struct ConstVariableProperties <: VariableProperties
    messageout
    marginal
end

function ConstVariableProperties(constant)
    messageout = of(Message(constant, true, false, nothing))
    marginal   = of(Marginal(constant, true, false, nothing))
    return ConstVariableProperties(messageout, marginal)
end

function setmessagein!(properties::ConstVariableProperties, messagein)
    # The `setmessagein!` function for the const variable is a no-op
    # because we do not pass any messages towards constants
    return properties, 1
end

# Old stuff is below

mutable struct ConstVariable{C, M} <: AbstractVariable
    name            :: Symbol
    collection_type :: AbstractVariableCollectionType
    constant        :: C
    messageout      :: M
    nconnected      :: Int
end

Base.show(io::IO, constvar::ConstVariable) = print(io, "ConstVariable(", indexed_name(constvar), ")")

"""
    constvar(value, [ dims... ])

Any runtime constant passed to a model as a model argument will be automatically converted 
to a fixed constant in the graph model at runtime. Sometimes it might be useful to create 
constants by hand (e.g. to avoid copying large matrices across the model and to 
avoid extensive memory allocations).

By default the `constvar` function wraps `Real` numbers and `AbstractArray` containers into 
the special `PointMass` structure, which represents the delta distribution centered around given value.

If the constant value is a `Function` the `constvar` function optionally accepts the `dims...` specification.
In case if `dims...` is not empty, the `constvar` function returns a container (a vector or a matrix, depending on the `dims...`)
with values computed by calling the provided function given the container indices as anrgument(s).
In case if `dims...` is empty, the `constvar` simply treats the function as fixed constant by itself.

Note: `constvar()` function is supposed to be used only within the `@model` macro.

## Example

```julia
@model function model_name(...)
    ...
    c = constvar(1.0)

    for i in 2:n
        x[i] ~ x[i - 1] + c # Reuse the same reference to a constant 1.0
    end
    ...
end
```
    
"""
function constvar end

constvar(name::Symbol, constval, collection_type::AbstractVariableCollectionType = VariableIndividual())                 = ConstVariable(name, collection_type, constval, of(Message(constval, true, false, nothing)), 0)
constvar(name::Symbol, constval::Real, collection_type::AbstractVariableCollectionType = VariableIndividual())           = constvar(name, PointMass(constval), collection_type)
constvar(name::Symbol, constval::AbstractArray, collection_type::AbstractVariableCollectionType = VariableIndividual())  = constvar(name, PointMass(constval), collection_type)
constvar(name::Symbol, constval::UniformScaling, collection_type::AbstractVariableCollectionType = VariableIndividual()) = constvar(name, PointMass(constval), collection_type)

function constvar(name::Symbol, fn::Function, length::Int)
    return map(i -> constvar(name, fn(i), VariableVector(i)), 1:length)
end

function constvar(name::Symbol, fn::Function, dim1::Int, dim2::Int, extra_dims::Vararg{Int})
    return constvar(name, fn, (dim1, dim2, extra_dims...))
end

function constvar(name::Symbol, fn::Function, dims::Tuple)
    indices = CartesianIndices(dims)
    size    = axes(indices)
    return map(i -> constvar(name, fn(convert(Tuple, i)), VariableArray(size, i)), indices)
end

degree(constvar::ConstVariable)          = nconnected(constvar)
name(constvar::ConstVariable)            = constvar.name
proxy_variables(constvar::ConstVariable) = nothing
collection_type(constvar::ConstVariable) = constvar.collection_type
setused!(constvar::ConstVariable)        = nothing

isproxy(::ConstVariable) = false

israndom(::ConstVariable)                  = false
israndom(::AbstractArray{<:ConstVariable}) = false
isdata(::ConstVariable)                    = false
isdata(::AbstractArray{<:ConstVariable})   = false
isconst(::ConstVariable)                   = true
isconst(::AbstractArray{<:ConstVariable})  = true

Base.getindex(constvar::ConstVariable, index) = Base.getindex(getconstant(constvar), index)

isconnected(constvar::ConstVariable) = constvar.nconnected !== 0
nconnected(constvar::ConstVariable)  = constvar.nconnected

getconst(constvar::ConstVariable{<:PointMass}) = BayesBase.getpointmass(constvar.constant)
getconst(constvar::ConstVariable)              = constvar.constant

getlastindex(::ConstVariable) = 1

messageout(constvar::ConstVariable, ::Int) = constvar.messageout
messagein(constvar::ConstVariable, ::Int)  = error("It is not possible to get a reference for inbound message for constvar")

get_pipeline_stages(::ConstVariable) = EmptyPipelineStage()

_getmarginal(constvar::ConstVariable)      = of(Marginal(constvar.constant, true, false, nothing))
_setmarginal!(::ConstVariable, observable) = error("It is not possible to set a marginal stream for `ConstVariable`")
_makemarginal(::ConstVariable)             = error("It is not possible to make marginal stream for `ConstVariable`")

setanonymous!(::ConstVariable, ::Bool) = nothing

function setmessagein!(constvar::ConstVariable, ::Int, messagein)
    constvar.nconnected += 1
    return nothing
end

# `ConstVariable` is the only one container that can 'undo' `as_variable` operation
undo_as_variable(constvar::ConstVariable) = getconst(constvar)
