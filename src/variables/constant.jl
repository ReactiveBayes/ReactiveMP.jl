export constvar

mutable struct ConstVariableProps
    messagein :: Union{Nothing, LazyObservable{AbstractMessage}}

    ConstVariableProps() = new(nothing)
end

struct ConstVariable{M} <: AbstractVariable
    name       :: Symbol
    messageout :: M
    props      :: ConstVariableProps
    belief     :: VariableBelief
end

constvar(name::Symbol, constval) = ConstVariable(name, of(Message(constval)), ConstVariableProps(), VariableBelief())

function messageout(constvar::ConstVariable, index::Int)
    @assert index === 1
    return constvar.messageout
end

function messagein(constvar::ConstVariable, index::Int)
    @assert index === 1
    return constvar.props.messagein
end

makebelief(constvar::ConstVariable)   = combineLatest((constvar.messageout, constvar.props.messagein), true, (AbstractBelief, reduce_message_to_belief))

function setmessagein!(constvar::ConstVariable, index::Int, messagein)
    @assert index === 1
    constvar.props.messagein = messagein
    return nothing
end
