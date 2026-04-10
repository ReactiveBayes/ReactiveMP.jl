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
TODO doc
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

function create_messagein!(randomvar::RandomVariable)
    messagein = MessageObservable(AbstractMessage)
    push!(randomvar.input_messages, messagein)
    return messagein, length(randomvar.input_messages)
end

function messagein(randomvar::RandomVariable, index::Int)
    return randomvar.input_messages[index]
end

function messageout(randomvar::RandomVariable, index::Int)
    return randomvar.output_messages[index]
end

struct RandomVariableActivationOptions{
    S, F <: MessageProductContext, M <: MessageProductContext
}
    scheduler::S
    prod_context_for_message_computation::F
    prod_context_for_marginal_computation::M
end

RandomVariableActivationOptions() = RandomVariableActivationOptions(
    AsapScheduler(), MessageProductContext(), MessageProductContext()
)

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
            schedule_on(options.scheduler),
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
