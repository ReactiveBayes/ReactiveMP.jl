export datavar

mutable struct DataVariableProps
    messagein :: Union{Nothing, LazyObservable{AbstractMessage}}

    DataVariableProps() = new(nothing)
end

struct DataVariable{S, D} <: AbstractVariable
    name       :: Symbol
    messageout :: S
    props      :: DataVariableProps
    belief     :: VariableBelief
end

function datavar(name::Symbol, ::Type{D}; subject::S = Subject(Message{D})) where { S, D }
    return DataVariable{S, D}(name, subject, DataVariableProps(), VariableBelief())
end

function messageout(datavar::DataVariable, index::Int)
    @assert index === 1
    return datavar.messageout
end

function messagein(datavar::DataVariable, index::Int)
    @assert index === 1
    return datavar.props.messagein
end

update!(datavar::DataVariable, data) = next!(messageout(datavar, 1), as_message(data))
finish!(datavar::DataVariable)       = complete!(messageout(datavar, 1))

makebelief(datavar::DataVariable)   = combineLatest((datavar.messageout, datavar.props.messagein), true, (AbstractBelief, reduce_message_to_belief))

function setmessagein!(datavar::DataVariable, index::Int, messagein)
    @assert index === 1
    datavar.props.messagein = messagein
    return nothing
end
