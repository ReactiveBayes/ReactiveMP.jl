export DataVariable, datavar, update!

import Base: show

mutable struct DataVariable{D, S} <: AbstractVariable
    name            :: Symbol
    collection_type :: AbstractVariableCollectionType
    messageout      :: S
    nconnected      :: Int
end

Base.show(io::IO, datavar::DataVariable) = print(io, "DataVariable(", indexed_name(datavar), ")")

function datavar(name::Symbol, ::Type{D}, collection_type::AbstractVariableCollectionType = VariableIndividual(); subject::S = RecentSubject(Union{Message{Missing}, Message{D}})) where { S, D }
    return DataVariable{D, S}(name, collection_type, subject, 0)
end

function datavar(name::Symbol, ::Type{D}, dims::Tuple; subject::S = RecentSubject(Union{Message{Missing}, Message{D}})) where { S, D }
    return datavar(name, D, dims...; subject = subject)
end

function datavar(name::Symbol, ::Type{D}, length::Int; subject::S = RecentSubject(Union{Message{Missing}, Message{D}})) where { S, D }
    vars = Vector{DataVariable{D, S}}(undef, length)
    @inbounds for i in 1:length
        vars[i] = datavar(name, D, VariableVector(i); subject = similar(subject))
    end
    return vars
end

function datavar(name::Symbol, ::Type{D}, dims::Vararg{Int}; subject::S = RecentSubject(Union{Message{Missing}, Message{D}})) where { S, D }
    vars = Array{DataVariable{D, S}}(undef, dims)
    @inbounds for i in CartesianIndices(axes(vars))
        vars[i] = datavar(name, D, VariableArray(i); subject = similar(subject))
    end
    return vars
end

Base.eltype(::Type{ <: DataVariable{D} }) where D = D
Base.eltype(::DataVariable{D})            where D = D

degree(datavar::DataVariable)          = nconnected(datavar)
name(datavar::DataVariable)            = datavar.name
collection_type(datavar::DataVariable) = datavar.collection_type
isconnected(datavar::DataVariable)     = datavar.nconnected !== 0
nconnected(datavar::DataVariable)      = datavar.nconnected

getlastindex(::DataVariable) = 1

messageout(datavar::DataVariable, ::Int) = datavar.messageout
messagein(datavar::DataVariable, ::Int)  = error("It is not possible to get a reference for inbound message for datavar")

update!(datavar::DataVariable, ::Missing)           = next!(messageout(datavar, 1), Message(missing, false, false))
update!(datavar::DataVariable, data::Number)        = update!(eltype(datavar), typeof(data), datavar, data)
update!(datavar::DataVariable, data::AbstractArray) = update!(eltype(datavar), typeof(data), datavar, data)

update!(::Type{ PointMass{D} }, ::Type{D}, datavar, data)   where { D }      = next!(messageout(datavar, 1), Message(PointMass(data), false, false))
update!(::Type{ PointMass{D1} }, ::Type{D2}, datavar, data) where { D1, D2 } = error("'$(name(datavar)) = datavar($D1, ...)' accepts data of type $D1, but $D2 has been supplied. Check 'update!($(name(datavar)), data::$D2)' and explicitly convert data to type $D1.")

resend!(datavar::DataVariable) = update!(datavar, Rocket.getrecent(messageout(datavar, 1)))

function update!(datavars::AbstractVector{ <: DataVariable }, data::AbstractVector)
    @assert size(datavars) === size(data) "Invalid update! call: size of datavar array and data should match"
    foreach(zip(datavars, data)) do (var, d)
        update!(var, d)
    end
end

finish!(datavar::DataVariable) = complete!(messageout(datavar, 1))

get_pipeline_stages(::DataVariable) = EmptyPipelineStage()

_getmarginal(datavar::DataVariable)              = datavar.messageout |> map(Marginal, as_marginal)
_setmarginal!(datavar::DataVariable, observable) = error("It is not possible to set a marginal stream for `DataVariable`")
_makemarginal(datavar::DataVariable)             = error("It is not possible to make marginal stream for `DataVariable`")

# Extension for _getmarginal
function Rocket.getrecent(proxy::ProxyObservable{ <: Marginal, S, M }) where { S <: Rocket.RecentSubjectInstance, D, M <: Rocket.MapProxy{D, typeof(as_marginal)} }
    return as_marginal(Rocket.getrecent(proxy.proxied_source))
end

function setmessagein!(datavar::DataVariable, ::Int, messagein)
    datavar.nconnected += 1
    return nothing
end