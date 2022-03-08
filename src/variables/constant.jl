export ConstVariable, constvar, getconst, isconnected

import Rocket: SingleObservable, AsapScheduler
import Base: getindex, show

mutable struct ConstVariable{C, M} <: AbstractVariable
    name            :: Symbol
    collection_type :: AbstractVariableCollectionType
    constant        :: C
    messageout      :: M
    nconnected      :: Int
end

Base.show(io::IO, constvar::ConstVariable) = print(io, "ConstVariable(", indexed_name(constvar), ")")

constvar(name::Symbol, constval, collection_type::AbstractVariableCollectionType = VariableIndividual())                 = ConstVariable(name, collection_type, constval, of(Message(constval, true, false)), 0)
constvar(name::Symbol, constval::Real, collection_type::AbstractVariableCollectionType = VariableIndividual())           = constvar(name, PointMass(constval), collection_type)
constvar(name::Symbol, constval::AbstractVector, collection_type::AbstractVariableCollectionType = VariableIndividual()) = constvar(name, PointMass(constval), collection_type)
constvar(name::Symbol, constval::AbstractMatrix, collection_type::AbstractVariableCollectionType = VariableIndividual()) = constvar(name, PointMass(constval), collection_type)

function constvar(name::Symbol, fn::Function, dims::Tuple)
    return constvar(name, fn, dims...)
end

function constvar(name::Symbol, fn::Function, length::Int)
    vars = Vector{ConstVariable}(undef, length)
    @inbounds for i in 1:length
        vars[i] = constvar(name, fn(i), VariableVector(i))
    end
    return vars
end

function constvar(name::Symbol, fn::Function, dims::Vararg{Int})
    vars = Array{ConstVariable}(undef, dims)
    size = axes(vars)
    @inbounds for i in CartesianIndices(axes(vars))
        vars[i] = constvar(name, fn(convert(Tuple, i)), VariableArray(size, i))
    end
    return vars
end

degree(constvar::ConstVariable)          = nconnected(constvar)
name(constvar::ConstVariable)            = constvar.name
proxy_variables(constvar::ConstVariable) = nothing
collection_type(constvar::ConstVariable) = constvar.collection_type

isproxy(::ConstVariable)  = false
israndom(::ConstVariable) = false

Base.getindex(constvar::ConstVariable, index) = Base.getindex(getconstant(constvar), index)

isconnected(constvar::ConstVariable) = constvar.nconnected !== 0
nconnected(constvar::ConstVariable)  = constvar.nconnected

getconst(constvar::ConstVariable{ <: PointMass }) = getpointmass(constvar.constant)
getconst(constvar::ConstVariable)                 = constvar.constant

getlastindex(::ConstVariable) = 1

messageout(constvar::ConstVariable, ::Int) = constvar.messageout
messagein(constvar::ConstVariable, ::Int)  = error("It is not possible to get a reference for inbound message for constvar")

get_pipeline_stages(::ConstVariable) = EmptyPipelineStage()

_getmarginal(constvar::ConstVariable)      = of(Marginal(constvar.constant, true, false))
_setmarginal!(::ConstVariable, observable) = error("It is not possible to set a marginal stream for `ConstVariable`")
_makemarginal(::ConstVariable)             = error("It is not possible to make marginal stream for `ConstVariable`")

# For _getmarginal
function Rocket.getrecent(observable::SingleObservable{ <: Marginal })
    return observable.value
end

function setmessagein!(constvar::ConstVariable, ::Int, messagein) 
    constvar.nconnected += 1
    return nothing
end
