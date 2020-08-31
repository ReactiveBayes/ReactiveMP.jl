export constvar

mutable struct ConstVariableProps
    messagein :: Union{Nothing, LazyObservable{Message}}
    marginal  :: Union{Nothing, MarginalObservable}

    ConstVariableProps() = new(nothing, nothing)
end

struct ConstVariable{M} <: AbstractVariable
    name       :: Symbol
    messageout :: M
    props      :: ConstVariableProps
end

constvar(name::Symbol, constval) = ConstVariable(name, of(Message(constval)), ConstVariableProps())

degree(::ConstVariable) = 1

getlastindex(::ConstVariable) = 1

function messageout(constvar::ConstVariable, index::Int)
    @assert index === 1
    return constvar.messageout
end

function messagein(constvar::ConstVariable, index::Int)
    @assert index === 1
    return constvar.props.messagein
end

_getmarginal(constvar::ConstVariable)                                = constvar.props.marginal
_setmarginal!(constvar::ConstVariable, marginal::MarginalObservable) = constvar.props.marginal = marginal
_makemarginal(constvar::ConstVariable)                               = constvar.messageout |> map(Marginal, as_marginal)

function setmessagein!(constvar::ConstVariable, index::Int, messagein)
    @assert index === 1 && constvar.props.messagein === nothing
    constvar.props.messagein = messagein
    return nothing
end
