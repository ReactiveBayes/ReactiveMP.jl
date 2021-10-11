export DataVariable, datavar, isconnected, update!

mutable struct DataVariable{D, S} <: AbstractVariable
    name       :: Symbol
    messageout :: S
    nconnected :: Int
    marginal   :: Union{Nothing, MarginalObservable}
end

function datavar(name::Symbol, ::Type{D}; subject::S = RecentSubject(Union{Message{Missing}, Message{D}})) where { S, D }
    return DataVariable{D, S}(name, subject, 0, nothing)
end

function datavar(name::Symbol, ::Type{D}, dims::Tuple; subject::S = RecentSubject(Union{Message{Missing}, Message{D}})) where { S, D }
    return datavar(name, D, dims...; subject = subject)
end

function datavar(name::Symbol, ::Type{D}, dims::Vararg{Int}; subject::S = RecentSubject(Union{Message{Missing}, Message{D}})) where { S, D }
    vars = Array{DataVariable{D, S}}(undef, dims)
    for i in CartesianIndices(axes(vars))
        @inbounds vars[i] = datavar(Symbol(name, :_, Symbol(join(i.I, :_))), D; subject = similar(subject))
    end
    return vars
end

Base.eltype(::Type{ <: DataVariable{D} }) where D = D
Base.eltype(::DataVariable{D})            where D = D

degree(datavar::DataVariable)    = nconnected(datavar)
name(datavar::DataVariable)      = datavar.name

isconnected(datavar::DataVariable) = datavar.nconnected !== 0
nconnected(datavar::DataVariable)  = datavar.nconnected

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

_getmarginal(datavar::DataVariable)                                = datavar.marginal
_setmarginal!(datavar::DataVariable, marginal::MarginalObservable) = datavar.marginal = marginal
_makemarginal(datavar::DataVariable)                               = datavar.messageout |> map(Marginal, as_marginal)

function setmessagein!(datavar::DataVariable, ::Int, messagein)
    datavar.nconnected += 1
    return nothing
end