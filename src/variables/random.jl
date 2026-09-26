export randomvar, RandomVariable, RandomVariableActivationOptions

## Random variable implementation

"""
    RandomVariable <: AbstractVariable

A latent variable, the quantity inference computes a posterior for. It has a stream of inbound
messages per connected factor node, as many outbound ones once it is activated, and the stream of
its marginal, the product of its inbound messages. Create one with [`randomvar`](@ref).

See also [`ConstVariable`](@ref), [`DataVariable`](@ref).
"""
mutable struct RandomVariable <: AbstractVariable
    input_messages::Vector{MessageObservable{AbstractMessage}}
    output_messages::Vector{MessageObservable{Message}}
    marginal::MarginalObservable
    label::Any
end

"""
    randomvar(; label = nothing) -> RandomVariable

Create a [`RandomVariable`](@ref), with no connections yet. `label` names it in the callback
events and in error messages, and is any value.

# Examples

```jldoctest
julia> x = randomvar(label = :x);

julia> ReactiveMP.israndom(x), ReactiveMP.degree(x)
(true, 0)
```
"""
function randomvar(; label = nothing)
    return RandomVariable(
        Vector{MessageObservable{AbstractMessage}}(),
        Vector{MessageObservable{Message}}(),
        MarginalObservable(),
        label,
    )
end

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
    RandomVariableActivationOptions(stream_postprocessor, prod_context_for_message_computation::MessageProductContext, prod_context_for_marginal_computation::MessageProductContext)
    RandomVariableActivationOptions()

What activating a [`RandomVariable`](@ref) needs, given positionally. With no arguments, no
postprocessor and two default [`ReactiveMP.MessageProductContext`](@ref)s.

# Fields

- `stream_postprocessor`: the stream postprocessor applied to the streams of the variable's
  [`ReactiveMP.EqualityChain`](@ref) and to its marginal stream, or `nothing` (see
  [`ReactiveMP.AbstractStreamPostprocessor`](@ref));
- `prod_context_for_message_computation`: how an outbound message, the product of every inbound
  message but the one on its own connection, is computed;
- `prod_context_for_marginal_computation`: how the marginal, the product of every inbound message,
  is computed. Its callbacks also receive [`ReactiveMP.BeforeMarginalComputationEvent`](@ref) and
  [`ReactiveMP.AfterMarginalComputationEvent`](@ref).

See also [`ReactiveMP.activate!`](@ref).
"""
struct RandomVariableActivationOptions{
        S, F <: MessageProductContext, M <: MessageProductContext,
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

Wire the streams of a random variable, after every factor node it connects to has been created.

1. **Outbound messages**: one stream per connection. With more than one connection, the
   message to connection `i` is the product of the inbound messages on every other connection,
   computed along a [`ReactiveMP.EqualityChain`](@ref), which reuses the partial products. With one
   connection, the outbound message never emits: there is nothing to multiply.
2. **Marginal**: the product of all inbound messages, formed with [`as_marginal`](@ref). It is
   computed once every inbound message has a value, and again whenever each has updated since.

A product that is initial does not consume its inputs, so the next update of any of them
computes it again.

# Throws

- `ArgumentError` for a variable with no connections.

See also [`RandomVariableActivationOptions`](@ref).
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
    return if is_initial(value)
        Rocket.fill_vstatus!(wrapper, true)
    end
end
