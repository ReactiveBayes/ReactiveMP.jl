export constvar, getconst, isconnected

import Rocket: SingleObservable, AsapScheduler
import Base: getindex

mutable struct ConstVariableProps
    nconnected :: Int

    ConstVariableProps() = new(0)
end

struct ConstVariable{C, M} <: AbstractVariable
    name       :: Symbol
    constant   :: C
    messageout :: M
    props      :: ConstVariableProps
end

constvar(name::Symbol, constval)                 = ConstVariable(name, constval, of(Message(constval, true, false)), ConstVariableProps())
constvar(name::Symbol, constval::Real)           = constvar(name, PointMass(constval))
constvar(name::Symbol, constval::AbstractVector) = constvar(name, PointMass(constval))
constvar(name::Symbol, constval::AbstractMatrix) = constvar(name, PointMass(constval))

function constvar(name::Symbol, fn::Function, dims::Tuple)
    return constvar(name, fn, dims...)
end

function constvar(name::Symbol, fn::Function, dims::Vararg{Int})
    vars = Array{ConstVariable}(undef, dims)
    for index in CartesianIndices(axes(vars))
        @inbounds vars[index] = constvar(Symbol(name, :_, Symbol(join(index.I, :_))), fn(convert(Tuple, index)))
    end
    return vars
end

degree(constvar::ConstVariable)     = nconnected(constvar)
name(constvar::ConstVariable)       = constvar.name
constraint(constvar::ConstVariable) = ClampedVariable()

Base.getindex(constvar::ConstVariable, index) = Base.getindex(getconstant(constvar), index)

isconnected(constvar::ConstVariable) = constvar.props.nconnected !== 0
nconnected(constvar::ConstVariable)  = constvar.props.nconnected

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
    constvar.props.nconnected += 1
    return nothing
end
