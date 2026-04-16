export randomvar, RandomVariable, RandomVariableActivationOptions

## Random variable implementation

"""
    RandomVariable <: AbstractVariable

Represents a latent (unobserved) variable in the factor graph. Random variables collect incoming and outgoing messages
from connected factor nodes and maintain a marginal belief. Use [`randomvar`](@ref) to create an instance.

See also: [`ReactiveMP.ConstVariable`](@ref), [`ReactiveMP.DataVariable`](@ref)
"""
mutable struct RandomVariable <: AbstractVariable
    input_messages  :: Vector{MessageObservable{AbstractMessage}}
    output_messages :: Vector{MessageObservable{Message}}
    marginal        :: MarginalObservable
    label           :: Any
end

"""
    randomvar(; label = nothing)

Creates a new [`ReactiveMP.RandomVariable`](@ref) with an optional `label` for identification.
"""
function randomvar(; label = nothing)
    return RandomVariable(
        Vector{MessageObservable{AbstractMessage}}(),
        Vector{MessageObservable{Message}}(),
        MarginalObservable(),
        label,
    )
end

"""
    ReactiveMP.degree(randomvar::RandomVariable)

Returns the number of factor nodes connected to `randomvar`, equal to the length of its inbound message streams collection.
See also [`ReactiveMP.degree`](@ref).
"""
degree(randomvar::RandomVariable) = length(randomvar.input_messages)

israndom(::RandomVariable) = true
israndom(::AbstractArray{<:RandomVariable}) = true
isdata(::RandomVariable) = false
isdata(::AbstractArray{<:RandomVariable}) = false
isconst(::RandomVariable) = false
isconst(::AbstractArray{<:RandomVariable}) = false

get_stream_of_marginals(randomvar::RandomVariable) = randomvar.marginal
get_stream_of_predictions(randomvar::RandomVariable) = randomvar.marginal

set_stream_of_marginals!(randomvar::RandomVariable, stream) =
    connect!(randomvar.marginal, stream)
set_stream_of_predictions!(randomvar::RandomVariable, stream) = error(
    "It is not possible to set a stream of predictions for `RandomVariable`"
)

function create_new_stream_of_inbound_messages!(randomvar::RandomVariable)
    new_stream_of_inbound_messages = MessageObservable(AbstractMessage)
    push!(randomvar.input_messages, new_stream_of_inbound_messages)
    return new_stream_of_inbound_messages, length(randomvar.input_messages)
end

function get_stream_of_inbound_messages(randomvar::RandomVariable, index::Int)
    return randomvar.input_messages[index]
end

function get_stream_of_outbound_messages(randomvar::RandomVariable, index::Int)
    return randomvar.output_messages[index]
end

"""
    RandomVariableActivationOptions

Collects all configuration needed to activate a [`ReactiveMP.RandomVariable`](@ref). Passed to [`ReactiveMP.activate!(::RandomVariable, ::RandomVariableActivationOptions)`](@ref).

Fields:
- `stream_postprocessor` — optional stream postprocessor applied to every created stream (see [`ReactiveMP.AbstractStreamPostprocessor`](@ref))
- `prod_context_for_message_computation` — a [`ReactiveMP.MessageProductContext`](@ref) used when computing outbound messages (product of all-but-one inbound messages in the `EqualityChain`)
- `prod_context_for_marginal_computation` — a [`ReactiveMP.MessageProductContext`](@ref) used when computing the marginal (product of all inbound messages)
"""
struct RandomVariableActivationOptions{
    S, F <: MessageProductContext, M <: MessageProductContext
}
    stream_postprocessor::S
    prod_context_for_message_computation::F
    prod_context_for_marginal_computation::M
end

RandomVariableActivationOptions() = RandomVariableActivationOptions(
    nothing, MessageProductContext(), MessageProductContext()
)

"""
    ReactiveMP.activate!(randomvar::RandomVariable, options::RandomVariableActivationOptions)

Wires all reactive streams of a [`ReactiveMP.RandomVariable`](@ref) into the factor graph.

Activation proceeds in two steps:

1. **Outbound messages** — resizes `output_messages` to match the number of connected nodes (the [`ReactiveMP.degree`](@ref)). If degree > 1, an `EqualityChain` is constructed: for each edge i the outbound message stream emits the product of all inbound messages *except* the one arriving on edge i, implementing the standard sum-product or variational update. If degree == 1 (a leaf variable), the single outbound stream is connected to `never()` because there are no other messages to multiply.

2. **Marginal** — `collectLatest` is called over all inbound [`ReactiveMP.MessageObservable`](@ref)s. It waits for all inbound messages to have emitted at least once, then emits the product as a new [`Marginal`](@ref) via [`ReactiveMP.set_stream_of_marginals!`](@ref), and re-emits only once all inbound messages have each updated again.

See also: [`ReactiveMP.RandomVariableActivationOptions`](@ref), [`ReactiveMP.activate!(::DataVariable, ::DataVariableActivationOptions)`](@ref)
"""
function activate!(
    randomvar::RandomVariable, options::RandomVariableActivationOptions
)
    d = length(randomvar.input_messages)
    outputmsgs = randomvar.output_messages
    resize!(outputmsgs, d)

    @inbounds for i in 1:d
        outputmsgs[i] = MessageObservable(Message)
    end

    if length(randomvar.input_messages) > 1
        chain = EqualityChain(
            randomvar.input_messages,
            options.stream_postprocessor,
            (messages) -> compute_product_of_messages(
                randomvar,
                options.prod_context_for_message_computation,
                messages,
            ),
        )
        initialize!(chain, outputmsgs)
    elseif length(randomvar.input_messages) == 1
        # If the number of input message is equal to `1`,
        # than the output message is not producing any value
        connect!(outputmsgs[1], never(Message))
    else
        throw(
            ArgumentError(
                "Cannot activate a random variable with zero or less than one inbound messages.",
            ),
        )
    end

    stream_of_marginals = collectLatest(
        AbstractMessage,
        Marginal,
        randomvar.input_messages,
        (messages) ->
            _compute_marginal_from_messages(randomvar, options, messages),
        reset_vstatus,
    )
    stream_of_marginals = postprocess_stream_of_marginals(
        options.stream_postprocessor, stream_of_marginals
    )

    set_stream_of_marginals!(randomvar, stream_of_marginals)

    return nothing
end

function _compute_marginal_from_messages(
    randomvar::RandomVariable,
    options::RandomVariableActivationOptions,
    messages,
)
    context = options.prod_context_for_marginal_computation
    span_id = generate_span_id(context.callbacks)
    invoke_callback(
        context.callbacks,
        BeforeMarginalComputationEvent(randomvar, context, messages, span_id),
    )
    result = as_marginal(
        compute_product_of_messages(randomvar, context, messages)
    )
    invoke_callback(
        context.callbacks,
        AfterMarginalComputationEvent(
            randomvar, context, messages, result, span_id
        ),
    )
    return result
end

# Reset consumption of the combination of inbound messages if the result of the computations is `is_initial`
# This is a helper function for the `EqualityChain` structure, but also for the marginals computation (both single and joint)
function reset_vstatus(wrapper, value)
    # We need to reset the internal Rocket.jl `vstatus` buffer from the `wrapper` if the result of the computation is `is_initial`
    # The `wrapper` here is the internal structure for `collectLatest` and `combineLatestUpdates` functions
    # The logic here is that if the result of the computation is `is_initial` we should reuse the arguments for the next computation
    # This may happen, when we initialize messages on the graph, which in turn also initializes marginals (implicitly)
    # if this happens, the inference cannot proceed further, since the initial messages have been consumed
    # This also prevents weird FE behaviour, when it "maximizes" the FE value, but converges to a minimum value
    if is_initial(value)
        Rocket.fill_vstatus!(wrapper, true)
    end
end
