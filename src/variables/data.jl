export datavar, DataVariable, new_observation!, DataVariableActivationOptions

"""
    DataVariable <: AbstractVariable

Represents an observed variable in the factor graph. Unlike [`ReactiveMP.ConstVariable`](@ref), the data is not fixed
at creation time and can be updated later via [`ReactiveMP.new_observation!`](@ref). Use [`datavar`](@ref) to create an instance.

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
get_stream_of_predictions(datavar::DataVariable) = datavar.prediction

set_stream_of_marginals!(datavar::DataVariable, stream) = connect!(
    datavar.marginal, stream
)
set_stream_of_predictions!(datavar::DataVariable, stream) = connect!(
    datavar.prediction, stream
)

function create_new_stream_of_inbound_messages!(datavar::DataVariable)
    new_stream_of_inbound_messages = MessageObservable(AbstractMessage)
    push!(datavar.input_messages, new_stream_of_inbound_messages)
    return new_stream_of_inbound_messages, length(datavar.input_messages)
end

function get_stream_of_inbound_messages(datavar::DataVariable, index::Int)
    return datavar.input_messages[index]
end

function get_stream_of_outbound_messages(datavar::DataVariable, ::Int)
    return datavar.messageout
end

"""
    DataVariableActivationOptions

Collects all configuration needed to activate a [`ReactiveMP.DataVariable`](@ref). Passed to [`ReactiveMP.activate!(::DataVariable, ::DataVariableActivationOptions)`](@ref).

Fields:
- `prediction::Bool` — if `true`, a prediction stream is built during activation as the product of all inbound (backward) messages
- `linked::Bool` — if `true`, the variable's observation stream is driven by a deterministic transformation of other variables' marginals rather than by direct [`ReactiveMP.new_observation!`](@ref) calls
- `transform` — the transformation function applied to the linked variables' marginals (used only when `linked = true`)
- `args` — the collection of linked variables or constants whose marginals are combined (used only when `linked = true`)
"""
struct DataVariableActivationOptions
    prediction::Bool
    linked::Bool
    transform
    args
end

DataVariableActivationOptions() = DataVariableActivationOptions(
    false, false, nothing, nothing
)

"""
    ReactiveMP.activate!(datavar::DataVariable, options::DataVariableActivationOptions)

Wires all reactive streams of a [`ReactiveMP.DataVariable`](@ref) into the factor graph.

Activation proceeds in up to three steps:

1. **Prediction** — if `options.prediction` is `true`, a prediction stream is built via `collectLatest` over all inbound (backward) [`ReactiveMP.MessageObservable`](@ref)s: once all backward messages have emitted and again when all of them update, their product is emitted as the model's prior expectation for this variable.

2. **Linked variables** — if `options.linked` is `true`, a subscription is created over a transformed combination of other variables' marginals. Each update is forwarded automatically to [`ReactiveMP.new_observation!`](@ref), making the data variable's observation a deterministic function of those variables.

3. **Marginal** — always wired: the marginal stream is `messageout |> map(as_marginal)`, so the marginal always equals the most recently pushed observation.

See also: [`ReactiveMP.DataVariableActivationOptions`](@ref), [`ReactiveMP.activate!(::RandomVariable, ::RandomVariableActivationOptions)`](@ref)
"""
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
        subscribe!(linkstream, (val) -> new_observation!(datavar, val))
    end

    # The marginal stream is always the same as the message out
    # but converted to Marginal with the as_marginal function
    stream_of_marginals = datavar.messageout |> map(Marginal, as_marginal)
    set_stream_of_marginals!(datavar, stream_of_marginals)

    return nothing
end

__link_getmarginal(constant) = of(Marginal(PointMass(constant), true, false))
__link_getmarginal(l::AbstractVariable) = get_stream_of_marginals(l)
__link_getmarginal(l::AbstractArray{<:AbstractVariable}) = collectLatest(
    map(get_stream_of_marginals, l)
)

__apply_link(f::F, args) where {F} = __apply_link(f, getdata.(args))
__apply_link(f::F, args::NTuple{N, PointMass}) where {F, N} = f(mean.(args)...)

"""
    new_observation!(datavar::DataVariable, data)
    new_observation!(datavars::AbstractArray{<:DataVariable}, data::AbstractArray)

Provides a new observation to a [`ReactiveMP.DataVariable`](@ref) (or an array of data variables).
The `data` is wrapped in a `PointMass` distribution and pushed as a new message.
Pass `missing` to indicate that the observation is not available.
"""
new_observation!(datavar::DataVariable, data) = new_observation!(
    datavar, PointMass(data)
)
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
