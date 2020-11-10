export constvar, getconstant, isconnected

mutable struct ConstVariableProps
    messagein :: Union{Nothing, LazyObservable{Message}}
    marginal  :: Union{Nothing, MarginalObservable}

    ConstVariableProps() = new(nothing, nothing)
end

struct ConstVariable{C} <: AbstractVariable
    name       :: Symbol
    constant   :: C
    props      :: ConstVariableProps
end

constvar(name::Symbol, constval)                 = ConstVariable(name, constval, ConstVariableProps())
constvar(name::Symbol, constval::Real)           = constvar(name, Dirac(constval))
constvar(name::Symbol, constval::AbstractVector) = constvar(name, Dirac(constval))
constvar(name::Symbol, constval::AbstractMatrix) = constvar(name, Dirac(constval))

degree(::ConstVariable) = 1

isconnected(constvar::ConstVariable) = constvar.props.messagein !== nothing

getconstant(constvar::ConstVariable) = constvar.constant
getlastindex(::ConstVariable) = 1

function messageout(constvar::ConstVariable, index::Int)
    @assert index === 1
    return of(as_message(constvar.constant))
end

function messagein(constvar::ConstVariable, index::Int)
    @assert index === 1
    return constvar.props.messagein
end

_getmarginal(constvar::ConstVariable)                                = constvar.props.marginal
_setmarginal!(constvar::ConstVariable, marginal::MarginalObservable) = constvar.props.marginal = marginal
_makemarginal(constvar::ConstVariable)                               = of(as_marginal(constvar.constant))

function setmessagein!(constvar::ConstVariable, index::Int, messagein)
    @assert index === 1 && constvar.props.messagein === nothing
    constvar.props.messagein = messagein
    return nothing
end
