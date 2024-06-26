export datavar, DataVariable, update!, DataVariableActivationOptions

mutable struct DataVariable{M, P} <: AbstractVariable
    input_messages :: Vector{MessageObservable{AbstractMessage}}
    marginal       :: MarginalObservable
    messageout     :: M
    prediction     :: P
end

function DataVariable()
    messageout = RecentSubject(Message)
    marginal   = MarginalObservable()
    prediction = MarginalObservable()
    return DataVariable(Vector{MessageObservable{AbstractMessage}}(), marginal, messageout, prediction)
end

datavar() = DataVariable()

degree(datavar::DataVariable) = length(datavar.input_messages)

israndom(::DataVariable)                  = false
israndom(::AbstractArray{<:DataVariable}) = false
isdata(::DataVariable)                    = true
isdata(::AbstractArray{<:DataVariable})   = true
isconst(::DataVariable)                   = false
isconst(::AbstractArray{<:DataVariable})  = false

function create_messagein!(datavar::DataVariable)
    messagein = MessageObservable(AbstractMessage)
    push!(datavar.input_messages, messagein)
    return messagein, length(datavar.input_messages)
end

function messagein(datavar::DataVariable, index::Int)
    return datavar.input_messages[index]
end

function messageout(datavar::DataVariable, ::Int)
    return datavar.messageout
end

struct DataVariableActivationOptions
    prediction::Bool
    linked::Bool
    transform
    args
end

DataVariableActivationOptions() = DataVariableActivationOptions(false, false, nothing, nothing)

function activate!(datavar::DataVariable, options::DataVariableActivationOptions)
    if options.prediction
        _setprediction!(datavar, _makeprediction(datavar))
    end

    if options.linked
        # If the variable is linked to another we need to apply a transformation from the linked variables
        # and redirect the updates to the `datavar` messageout stream
        linkvalues = combineLatestUpdates(map(l -> __link_getmarginal(l), options.args))
        linkstream = linkvalues |> map(Any, (args) -> let f = options.transform
            return __apply_link(f, getrecent.(args))
        end)
        # This subscription should unsubscribe automatically when the linked `datavar`s complete
        subscribe!(linkstream, (val) -> update!(datavar, val))
    end

    # The marginal stream is always the same as the message out
    connect!(datavar.marginal, datavar.messageout |> map(Marginal, as_marginal))

    return nothing
end

__link_getmarginal(constant) = of(Marginal(PointMass(constant), true, false, nothing))
__link_getmarginal(l::AbstractVariable) = getmarginal(l, IncludeAll())
__link_getmarginal(l::AbstractArray{<:AbstractVariable}) = getmarginals(l, IncludeAll())

__apply_link(f::F, args) where {F} = __apply_link(f, getdata.(args))
__apply_link(f::F, args::NTuple{N, PointMass}) where {F, N} = f(mean.(args)...)

_getmarginal(datavar::DataVariable)       = datavar.marginal
_setmarginal!(::DataVariable, observable) = error("It is not possible to set a marginal stream for `DataVariable`")
_makemarginal(::DataVariable)             = error("It is not possible to make marginal stream for `DataVariable`")

update!(datavar::DataVariable, data)            = update!(datavar, PointMass(data))
update!(datavar::DataVariable, data::PointMass) = next!(datavar.messageout, Message(data, false, false, nothing))
update!(datavar::DataVariable, ::Missing)       = next!(datavar.messageout, Message(missing, false, false, nothing))

function update!(datavars::AbstractArray{<:DataVariable}, data::AbstractArray)
    @assert size(datavars) === size(data) """
    Invalid `update!` call: size of datavar array and data must match: `variables` has size $(size(datavars)) and `data` has size $(size(data)). 
    """
    foreach(zip(datavars, data)) do (var, d)
        update!(var, d)
    end
end

function update!(datavars::AbstractArray{<:DataVariable}, data::Missing)
    foreach(datavars) do var
        update!(var, data)
    end
end

marginal_prod_fn(datavar::DataVariable) = marginal_prod_fn(FoldLeftProdStrategy(), GenericProd(), UnspecifiedFormConstraint(), FormConstraintCheckLast())

_getprediction(datavar::DataVariable)              = datavar.prediction
_setprediction!(datavar::DataVariable, observable) = connect!(_getprediction(datavar), observable)
_makeprediction(datavar::DataVariable)             = collectLatest(AbstractMessage, Marginal, datavar.input_messages, marginal_prod_fn(datavar))
