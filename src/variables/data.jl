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

set_stream_of_marginals!(datavar::DataVariable, stream) =
    connect!(datavar.marginal, stream)
set_stream_of_predictions!(datavar::DataVariable, stream) =
    connect!(datavar.prediction, stream)

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

DataVariableActivationOptions() =
    DataVariableActivationOptions(false, false, nothing, nothing)

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
__link_getmarginal(l::AbstractArray{<:AbstractVariable}) =
    collectLatest(map(get_stream_of_marginals, l))

__apply_link(f::F, args) where {F} = __apply_link_data(f, getdata.(args))

__apply_link_data(f::F, data::NTuple{N, PointMass}) where {F, N} =
    f(mean.(data)...)

# A linked `DataVariable` must be a deterministic function of *point* values: the
# transformation is applied to plain numbers, not to distributions. Linking to a
# `RandomVariable` therefore delivers a full posterior here, which has no meaningful
# point value to substitute. Previously this produced a bare `MethodError` mentioning
# only `__apply_link`, which gives no indication of what the user did wrong.
function __apply_link_data(f::F, data::Tuple) where {F}
    offenders = join(
        (
            "  argument $(i) :: $(typeof(d))" for
            (i, d) in enumerate(data) if !(d isa PointMass)
        ),
        "\n",
    )
    error(
        """
        Cannot apply the link function `$(f)` to a linked data variable: every linked argument must resolve to a `PointMass`, but the following did not:
        $(offenders)

        A linked data variable is a deterministic function of observed point values, so its arguments must be constants or other data variables holding observations. Linking to a random variable is not supported, because its marginal is a distribution rather than a point value.

        If you intended to use the random variable's expectation, link to a data variable that you update explicitly with `new_observation!`, or introduce a deterministic node into the model instead.""",
    )
end

"""
    new_observation!(datavar::DataVariable, data)
    new_observation!(datavars::AbstractArray{<:DataVariable}, data::AbstractArray)

Provides a new observation to a [`ReactiveMP.DataVariable`](@ref) (or an array of data variables).
The `data` is wrapped in a `PointMass` distribution and pushed as a new message.
Pass `missing` to indicate that the observation is not available.

The value must be a real number, an array of real numbers or a `UniformScaling` — the payloads
for which `PointMass` defines a `variate_form`, and hence a usable `mean`. Anything else is
rejected with an informative error. An observation of a different kind, such as text consumed by
a custom node, has to be wrapped in a `PointMass` explicitly; that method performs no validation.
See [Non-standard observations](@ref lib-variables-data-nonstandard).
"""
function new_observation!(datavar::DataVariable, data)
    __assert_valid_observation(datavar, data)
    return new_observation!(datavar, PointMass(data))
end
new_observation!(datavar::DataVariable, data::PointMass) = next!(datavar.messageout, Message(data, false, false))
new_observation!(datavar::DataVariable, ::Missing)       = next!(datavar.messageout, Message(missing, false, false))

# `PointMass` only defines `variate_form` (and hence usable `mean`/`var`) for these payloads.
# Wrapping anything else produces a `PointMass` that *constructs* fine but whose `mean` recurses
# between `Statistics.mean(itr)` and `BayesBase.mean(fn, ::PointMass)` until the stack overflows,
# which surfaces tens of thousands of frames deep with no hint of the actual mistake (issue #588).
__assert_valid_observation(
    ::DataVariable, ::Union{Real, AbstractArray, UniformScaling}
) = nothing

# A non-numeric payload is not always a mistake. A custom node may dispatch its rules on, say,
# `PointMass{<:String}` and read the payload back with `getpointmass`, never calling `mean` — for
# which the explicitly-wrapped `new_observation!(::DataVariable, ::PointMass)` method, deliberately
# left unvalidated, is the supported route. The error points at it, since being told to pass a real
# number is unhelpful when the observation genuinely is not one.
#
# Distributions are excluded from that hint: `PointMass(Beta(1, 1))` is not what anyone means, so
# suggesting the wrap there would only route a real mistake around the guard.
function __observation_hint(::Type{D}, varname) where {D <: Distribution}
    return """
    Passing a distribution as data is not supported: data variables hold observed point values, not beliefs. To place a prior on a quantity, make it a random variable in the model instead."""
end

function __observation_hint(::Type{D}, varname) where {D}
    return """
    If the value is *intentionally* not numeric — for example text consumed by a custom node whose rules dispatch on `PointMass{<:$(D)}` — wrap it in a `PointMass` yourself:

        new_observation!($(varname), PointMass(value))

    `new_observation!(::DataVariable, ::PointMass)` performs no validation, so the payload reaches the connected factor nodes untouched. In exchange, such a `PointMass` has no `variate_form`, and therefore no `mean`, `var` or `logpdf`: only rules that dispatch on its concrete type and read the payload with `BayesBase.getpointmass` can consume it. See the "Non-standard observations" section of the ReactiveMP.jl documentation."""
end

function __assert_valid_observation(datavar::DataVariable, data::D) where {D}
    label = something(datavar.label, "")
    named = isempty(string(label)) ? "" : " for `$(label)`"
    varname = isempty(string(label)) ? "y" : string(label)
    error(
        """
        Invalid observation$(named): `$(D)` cannot be used as observed data.

        Observations must be a real number, an array of real numbers, or a `UniformScaling`. Got a value of type `$(D)`.

        If you meant to indicate that this observation is not available, pass `missing` instead.

        $(__observation_hint(D, varname))""",
    )
end

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
