export datavar, isconnected

mutable struct DataVariableProps
    nconnected :: Int

    DataVariableProps() = new(0)
end

struct DataVariable{S, D} <: AbstractVariable
    name       :: Symbol
    messageout :: S
    props      :: DataVariableProps
end

function datavar(name::Symbol, ::Type{D}; subject::S = Subject(Message{D})) where { S, D }
    return DataVariable{S, D}(name, subject, DataVariableProps())
end

function datavar(name::Symbol, ::Type{D}, dims::Tuple; subject::S = Subject(Message{D})) where { S, D }
    return datavar(name, D, dims...; subject = subject)
end

function datavar(name::Symbol, ::Type{D}, dims::Vararg{Int}; subject::S = Subject(Message{D})) where { S, D }
    vars = Array{DataVariable{S, D}}(undef, dims)
    for index in CartesianIndices(axes(vars))
        @inbounds vars[index] = datavar(Symbol(name, :_, Symbol(join(index.I, :_))), D; subject = similar(subject))
    end
    return vars
end

degree(datavar::DataVariable) = nconnected(datavar)
name(datavar::DataVariable)   = datavar.name

isconnected(datavar::DataVariable) = datavar.props.nconnected !== 0
nconnected(datavar::DataVariable)  = datavar.props.nconnected

getlastindex(::DataVariable) = 1

messageout(datavar::DataVariable, ::Int) = datavar.messageout
messagein(datavar::DataVariable, ::Int)  = error("It is not possible to get a reference for inbound message for datavar")

update!(datavar::DataVariable, data)                 = next!(messageout(datavar, 1), as_message(data))
update!(datavar::DataVariable, data::Real)           = next!(messageout(datavar, 1), as_message(Dirac(data)))
update!(datavar::DataVariable, data::AbstractVector) = next!(messageout(datavar, 1), as_message(Dirac(data)))
update!(datavar::DataVariable, data::AbstractMatrix) = next!(messageout(datavar, 1), as_message(Dirac(data)))

finish!(datavar::DataVariable) = complete!(messageout(datavar, 1))

_getmarginal(datavar::DataVariable)                                = datavar.messageout |> map(Marginal, as_marginal)
_setmarginal!(datavar::DataVariable, marginal::MarginalObservable) = error("It is not possible to set a marginal stream for datavar")
_makemarginal(datavar::DataVariable)                               = error("It is not possible to make marginal stream for datavar")

function setmessagein!(datavar::DataVariable, ::Int, messagein)
    datavar.props.nconnected += 1
    return nothing
end