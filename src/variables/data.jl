export datavar

mutable struct DataVariableProps
    messagein :: Union{Nothing, LazyObservable{AbstractMessage}}

    DataVariableProps() = new(nothing)
end

struct DataVariable{S, D} <: AbstractVariable
    name       :: Symbol
    messageout :: S
    props      :: DataVariableProps
    marginal   :: VariableMarginal
end

function datavar(name::Symbol, ::Type{D}; subject::S = Subject(Message{D})) where { S, D }
    return DataVariable{S, D}(name, subject, DataVariableProps(), VariableMarginal())
end

degree(::DataVariable) = 1

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

makemarginal(datavar::DataVariable)   = combineLatest(datavar.messageout, datavar.props.messagein, strategy = PushNew()) |> reduce_to_marginal

function setmessagein!(datavar::DataVariable, index::Int, messagein)
    @assert index === 1
    datavar.props.messagein = messagein
    return nothing
end
