export constvar

mutable struct ConstVariableProps
    messagein :: Union{Nothing, LazyObservable{AbstractMessage}}

    ConstVariableProps() = new(nothing)
end

struct ConstVariable{M} <: AbstractVariable
    name       :: Symbol
    messageout :: M
    props      :: ConstVariableProps
    marginal   :: VariableMarginal
end

constvar(name::Symbol, constval) = ConstVariable(name, of(Message(constval)), ConstVariableProps(), VariableMarginal())

degree(::ConstVariable) = 1

function messageout(constvar::ConstVariable, index::Int)
    @assert index === 1
    return constvar.messageout
end

function messagein(constvar::ConstVariable, index::Int)
    @assert index === 1
    return constvar.props.messagein
end

makemarginal(constvar::ConstVariable) = combineLatest(constvar.messageout, constvar.props.messagein, strategy = PushNew()) |> reduce_to_marginal

function setmessagein!(constvar::ConstVariable, index::Int, messagein)
    @assert index === 1
    constvar.props.messagein = messagein
    return nothing
end
