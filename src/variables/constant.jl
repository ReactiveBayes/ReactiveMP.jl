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

constvar(name::Symbol, constval)                 = ConstVariable(name, constval, of(as_message(constval)), ConstVariableProps())
constvar(name::Symbol, constval::Real)           = constvar(name, Dirac(constval))
constvar(name::Symbol, constval::AbstractVector) = constvar(name, Dirac(constval))
constvar(name::Symbol, constval::AbstractMatrix) = constvar(name, Dirac(constval))

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

degree(constvar::ConstVariable) = nconnected(constvar)
name(constvar::ConstVariable)   = constvar.name

Base.getindex(constvar::ConstVariable, index) = Base.getindex(getconstant(constvar), index)

isconnected(constvar::ConstVariable) = constvar.props.nconnected !== 0
nconnected(constvar::ConstVariable)  = constvar.props.nconnected

getconst(constvar::ConstVariable{ <: Dirac }) = getpointmass(constvar.constant)
getconst(constvar::ConstVariable)             = constvar.constant

getlastindex(::ConstVariable) = 1

messageout(constvar::ConstVariable, ::Int) = constvar.messageout
messagein(constvar::ConstVariable, ::Int)  = error("It is not possible to get a reference for inbound message for constvar")

_getmarginal(constvar::ConstVariable) = of(as_marginal(constvar.constant))

_setmarginal!(::ConstVariable, ::MarginalObservable) = error("It is not possible to set a marginal stream for constvar")
_makemarginal(::ConstVariable)                       = error("It is not possible to make marginal stream for constvar")

function setmessagein!(constvar::ConstVariable, ::Int, messagein) 
    constvar.props.nconnected += 1
    return nothing
end
