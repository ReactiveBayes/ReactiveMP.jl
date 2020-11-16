export constvar, getconstant, isconnected

import Rocket: SingleObservable, AsapScheduler

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

degree(constvar::ConstVariable) = nconnected(constvar)
name(constvar::ConstVariable)   = constvar.name

isconnected(constvar::ConstVariable) = constvar.props.nconnected !== 0
nconnected(constvar::ConstVariable)  = constvar.props.nconnected

getconstant(constvar::ConstVariable) = constvar.constant
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
