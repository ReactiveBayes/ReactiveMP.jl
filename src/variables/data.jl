export datavar, DataVariable, new_observation!, DataVariableActivationOptions

"""
    DataVariable <: AbstractVariable

An observed variable, whose value arrives after the graph is built, with
[`new_observation!`](@ref), and may change: each observation propagates through the graph. Every
connected node shares its one outbound stream, of the observations, and has a stream of the
messages it sends back, whose product is the variable's prediction. Its marginal is its latest
observation. Create one with [`datavar`](@ref).

See also [`RandomVariable`](@ref), [`ConstVariable`](@ref).
"""
mutable struct DataVariable{M, P} <: AbstractVariable
    input_messages::Vector{MessageObservable{AbstractMessage}}
    marginal::MarginalObservable
    messageout::M
    prediction::P
    label::Any
end

function DataVariable(; label = nothing)
    messageout = RecentSubject(Message)
    marginal = MarginalObservable()
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
    datavar(; label = nothing) -> DataVariable

Create a [`DataVariable`](@ref), with no observation and no connections yet. `label` names it in
the callback events and in the error of an invalid observation.

# Examples

```jldoctest
julia> y = datavar(label = :y);

julia> ReactiveMP.isdata(y), ReactiveMP.degree(y)
(true, 0)
```
"""
datavar(; label = nothing) = DataVariable(; label = label)

degree(datavar::DataVariable) = length(datavar.input_messages)

israndom(::DataVariable) = false
israndom(::AbstractArray{<:DataVariable}) = false
isdata(::DataVariable) = true
isdata(::AbstractArray{<:DataVariable}) = true
isconst(::DataVariable) = false
isconst(::AbstractArray{<:DataVariable}) = false

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
    DataVariableActivationOptions(prediction::Bool, linked::Bool, transform, args)
    DataVariableActivationOptions()

What activating a [`DataVariable`](@ref) needs, given positionally. With no arguments, no
prediction and no link.

# Fields

- `prediction`: whether to build the prediction stream, the product of the messages the nodes
  send to the variable (see [`ReactiveMP.get_stream_of_predictions`](@ref));
- `linked`: whether the variable's observations are a function of other variables, rather than
  given with [`new_observation!`](@ref);
- `transform`: for a linked variable, the function its observation is of;
- `args`: for a linked variable, the constants and variables `transform` takes, in order. Each
  must resolve to a `PointMass`: a constant, or a data variable holding an observation.

See also [`ReactiveMP.activate!`](@ref).
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

Wire the streams of a data variable, after every factor node it connects to has been created.

1. **Prediction**, with `options.prediction`: the product of the messages the nodes send to the
   variable, computed once every one has a value and again whenever each has updated since.
2. **Link**, with `options.linked`: whenever each of `options.args` has a value, and again when
   one updates, `options.transform` of their values is observed with [`new_observation!`](@ref).
   Linking to a random variable fails at the first update, since its marginal is not a point
   value.
3. **Marginal**: always the latest observation, as a [`Marginal`](@ref).

See also [`DataVariableActivationOptions`](@ref).
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
            linkvalues |> map(
            Any, (args) -> let f = options.transform
                return __apply_link(f, getrecent.(args))
            end
        )
        # This subscription should unsubscribe automatically when the linked `datavar`s complete
        subscribe!(linkstream, (val) -> new_observation!(datavar, val))
    end

    # The marginal stream is always the same as the message out
    # but converted to Marginal with the as_marginal function
    stream_of_marginals = datavar.messageout |> map(Marginal, as_marginal)
    set_stream_of_marginals!(datavar, stream_of_marginals)

    return nothing
end

__link_getmarginal(constant) = of(Marginal(PointMass(constant), true, false, AnnotationDict(), 0))
__link_getmarginal(l::AbstractVariable) = get_stream_of_marginals(l)
__link_getmarginal(l::AbstractArray{<:AbstractVariable}) =
    collectLatest(map(get_stream_of_marginals, l))

__apply_link(f::F, args) where {F} = __apply_link_data(f, getdata.(args))

__apply_link_data(f::F, data::NTuple{N, PointMass}) where {F, N} =
    f(mean.(data)...)

# A linked `DataVariable` must be a deterministic function of *point* values: the
# transformation is applied to plain numbers, not to distributions. Linking to a
# `RandomVariable` therefore delivers a full posterior here, which has no meaningful
# point value to substitute; the error says so, where a bare `MethodError` in
# `__apply_link` would not.
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
    new_observation!(datavar::DataVariable, data::PointMass)
    new_observation!(datavar::DataVariable, ::Missing)
    new_observation!(datavars::AbstractArray{<:DataVariable}, data::AbstractArray)
    new_observation!(datavars::AbstractArray{<:DataVariable}, ::Missing)

Observe `data` on a data variable, and propagate it through the graph: the variable sends it to
its nodes as a `PointMass` message, with log scale zero.

- A real number, an array of real numbers or a `UniformScaling` is wrapped in a `PointMass`:
  these are the values for which `PointMass` defines a `variate_form`, and so a `mean`.
- A `PointMass` is sent as it is, without validation: this is how an observation of another
  kind, such as text a custom node reads with `BayesBase.getpointmass`, is given. See
  [Non-standard observations](@ref lib-variables-data-nonstandard).
- `missing` says the observation is not available: the message is `missing`, and the rules
  that depend on it give `missing` in turn.
- An array of data variables observes an array of values of the same size, element by element,
  or `missing` on every variable.

# Throws

- `ErrorException` for any other value, a distribution included, naming the variable and the
  type;
- `AssertionError` when an array of data variables and an array of values differ in size.

# Examples

```jldoctest
julia> y = datavar();

julia> new_observation!(y, 1.5)

julia> new_observation!([datavar(), datavar()], [1.0, 2.0])
```
"""
function new_observation!(datavar::DataVariable, data)
    __assert_valid_observation(datavar, data)
    return new_observation!(datavar, PointMass(data))
end
# An observation is a point mass: its log scale is zero.
new_observation!(datavar::DataVariable, data::PointMass) = next!(datavar.messageout, Message(data, false, false, AnnotationDict(), 0))
new_observation!(datavar::DataVariable, ::Missing) = next!(datavar.messageout, Message(missing, false, false))

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
    return foreach(zip(datavars, data)) do (var, d)
        new_observation!(var, d)
    end
end

function new_observation!(
        datavars::AbstractArray{<:DataVariable}, data::Missing
    )
    return foreach(datavars) do var
        new_observation!(var, data)
    end
end
