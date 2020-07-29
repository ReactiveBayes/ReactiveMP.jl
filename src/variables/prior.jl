export priorvar

mutable struct PriorVariableProps
    messagein :: Union{Nothing, LazyObservable{Message}}

    PriorVariableProps() = new(nothing)
end

struct PriorVariable{S, D} <: AbstractVariable
    name       :: Symbol
    messageout :: S
    props      :: PriorVariableProps
    marginal   :: VariableMarginal
end

function priorvar(name::Symbol, ::Type{D}; subject::S = Subject(Message{D})) where { S, D }
    return PriorVariable{S, D}(name, subject, PriorVariableProps(), VariableMarginal())
end

degree(::PriorVariable) = 1

function messageout(priorvar::PriorVariable, index::Int)
    @assert index === 1
    return priorvar.messageout
end

function messagein(priorvar::PriorVariable, index::Int)
    @assert index === 1
    return priorvar.props.messagein
end

update!(priorvar::PriorVariable, data) = next!(messageout(priorvar, 1), as_message(data))
finish!(priorvar::PriorVariable)       = complete!(messageout(priorvar, 1))

makemarginal(priorvar::PriorVariable)   = combineLatest(priorvar.messageout, priorvar.props.messagein, strategy = PushNewBut{1}()) |> reduce_to_marginal

function setmessagein!(priorvar::PriorVariable, index::Int, messagein)
    @assert index === 1
    priorvar.props.messagein = messagein
    return nothing
end
