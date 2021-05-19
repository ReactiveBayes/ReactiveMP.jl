export ConstVariable, constvar, getconst, isconnected

import Rocket: SingleObservable, AsapScheduler
import Base: getindex

mutable struct ConstVariable{C, M} <: AbstractVariable
    name       :: Symbol
    constant   :: C
    messageout :: M
    nconnected :: Int
end

constvar(name::Symbol, constval)                 = ConstVariable(name, constval, of(Message(constval, true, false)), 0)
constvar(name::Symbol, constval::Real)           = constvar(name, PointMass(constval))
constvar(name::Symbol, constval::AbstractVector) = constvar(name, PointMass(constval))
constvar(name::Symbol, constval::AbstractMatrix) = constvar(name, PointMass(constval))

function constvar(name::Symbol, fn::Function, dims::Tuple)
    return constvar(name, fn, dims...)
end

function constvar(name::Symbol, fn::Function, dims::Vararg{Int})
    vars = Array{ConstVariable}(undef, dims)
    for i in CartesianIndices(axes(vars))
        @inbounds vars[i] = constvar(Symbol(name, :_, Symbol(join(i.I, :_))), fn(convert(Tuple, i)))
    end
    return vars
end

degree(constvar::ConstVariable)     = nconnected(constvar)
name(constvar::ConstVariable)       = constvar.name
local_constraint(::ConstVariable)   = ClampedVariable()

Base.getindex(constvar::ConstVariable, index) = Base.getindex(getconstant(constvar), index)

isconnected(constvar::ConstVariable) = constvar.nconnected !== 0
nconnected(constvar::ConstVariable)  = constvar.nconnected

getconst(constvar::ConstVariable{ <: PointMass }) = getpointmass(constvar.constant)
getconst(constvar::ConstVariable)                 = constvar.constant

getlastindex(::ConstVariable) = 1

messageout(constvar::ConstVariable, ::Int) = constvar.messageout
messagein(constvar::ConstVariable, ::Int)  = error("It is not possible to get a reference for inbound message for constvar")

inbound_portal(::ConstVariable) = EmptyPortal()

_getmarginal(constvar::ConstVariable) = of(Marginal(constvar.constant, true, false))

_setmarginal!(::ConstVariable, ::MarginalObservable) = error("It is not possible to set a marginal stream for constvar")
_makemarginal(::ConstVariable)                       = error("It is not possible to make marginal stream for constvar")

function setmessagein!(constvar::ConstVariable, ::Int, messagein) 
    constvar.nconnected += 1
    return nothing
end
