export datavar, DataVariable, new_observation!, DataVariableActivationOptions

"""
    DataVariable <: AbstractVariable

Represents an observed variable in the factor graph. Unlike [`ReactiveMP.ConstVariable`](@ref), the data is not fixed
at creation time and can be updated later via [`update!`](@ref). Use [`datavar`](@ref) to create an instance.

See also: [`ReactiveMP.RandomVariable`](@ref), [`ReactiveMP.ConstVariable`](@ref)
"""
mutable struct DataVariable{M, P} <: AbstractVariable
    input_messages :: Vector{MessageObservable{AbstractMessage}}
    marginal       :: MarginalObservable
    messageout     :: M
    prediction     :: P
    label          :: Any
end

function DataVariable(; label = nothing)
    messageout = RecentSubject(Message)
    marginal   = MarginalObservable()
    prediction = MarginalObservable()
    return DataVariable(
        Vector{MessageObservable{AbstractMessage}}(),
        marginal,
        messageout,
        prediction,
        label,
    )
end

"""
    datavar(; label = nothing)

Creates a new [`ReactiveMP.DataVariable`](@ref) with an optional `label` for identification.
"""
datavar(; label = nothing) = DataVariable(; label = label)

degree(datavar::DataVariable) = length(datavar.input_messages)

israndom(::DataVariable)                  = false
israndom(::AbstractArray{<:DataVariable}) = false
isdata(::DataVariable)                    = true
isdata(::AbstractArray{<:DataVariable})   = true
isconst(::DataVariable)                   = false
isconst(::AbstractArray{<:DataVariable})  = false

get_stream_of_marginals(datavar::DataVariable) = datavar.marginal

set_stream_of_marginals!(datavar::DataVariable, stream) =
    connect!(datavar.marginal, stream)
set_stream_of_predictions!(datavar::DataVariable, stream) =
    connect!(datavar.prediction, stream)

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

DataVariableActivationOptions() =
    DataVariableActivationOptions(false, false, nothing, nothing)

function activate!(
    datavar::DataVariable, options::DataVariableActivationOptions
)
    if options.prediction
        # if the prediction is requested, we instantiate the stream of predictions 
        # as the product of all inbound messages to the datavar 
        # otherwise the stream of predictions is empty
        stream_of_predictions = collectLatest(
            AbstractMessage,
            Marginal,
            datavar.input_messages,
            (messages) -> as_marginal(
                compute_product_of_messages(
                    datavar, MessageProductContext(), messages
                ),
            ),
        )
        set_stream_of_predictions!(datavar, stream_of_predictions)
    end

    if options.linked
        # If the variable is linked to another we need to apply a transformation from the linked variables
        # and redirect the updates to the `datavar` messageout stream
        linkvalues = combineLatestUpdates(
            map(l -> __link_getmarginal(l), options.args)
        )
        linkstream =
            linkvalues |> map(Any, (args) -> let f = options.transform
                return __apply_link(f, getrecent.(args))
            end)
        # This subscription should unsubscribe automatically when the linked `datavar`s complete
        subscribe!(linkstream, (val) -> update!(datavar, val))
    end

    # The marginal stream is always the same as the message out
    # but converted to Marginal with the as_marginal function
    stream_of_marginals = datavar.messageout |> map(Marginal, as_marginal)
    set_stream_of_marginals!(datavar, stream_of_marginals)

    return nothing
end

__link_getmarginal(constant) = of(Marginal(PointMass(constant), true, false))
__link_getmarginal(l::AbstractVariable) = get_stream_of_marginals(l)
__link_getmarginal(l::AbstractArray{<:AbstractVariable}) =
    collectLatest(map(get_stream_of_marginals, l))

__apply_link(f::F, args) where {F} = __apply_link(f, getdata.(args))
__apply_link(f::F, args::NTuple{N, PointMass}) where {F, N} = f(mean.(args)...)

"""
    new_observation!(datavar::DataVariable, data)
    new_observation!(datavars::AbstractArray{<:DataVariable}, data::AbstractArray)

Provides a new observation to a [`ReactiveMP.DataVariable`](@ref) (or an array of data variables).
The `data` is wrapped in a `PointMass` distribution and pushed as a new message.
Pass `missing` to indicate that the observation is not available.
"""
new_observation!(datavar::DataVariable, data) =
    new_observation!(datavar, PointMass(data))
new_observation!(datavar::DataVariable, data::PointMass) = next!(datavar.messageout, Message(data, false, false))
new_observation!(datavar::DataVariable, ::Missing)       = next!(datavar.messageout, Message(missing, false, false))

function new_observation!(
    datavars::AbstractArray{<:DataVariable}, data::AbstractArray
)
    @assert size(datavars) === size(data) """
    Invalid `new_observation!` call: size of datavar array and data must match: `variables` has size $(size(datavars)) and `data` has size $(size(data)). 
    """
    foreach(zip(datavars, data)) do (var, d)
        new_observation!(var, d)
    end
end

function new_observation!(
    datavars::AbstractArray{<:DataVariable}, data::Missing
)
    foreach(datavars) do var
        new_observation!(var, data)
    end
end
