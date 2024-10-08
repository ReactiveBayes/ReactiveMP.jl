export randomvar, RandomVariable, RandomVariableActivationOptions

## Random variable implementation

mutable struct RandomVariable <: AbstractVariable
    input_messages::Vector{MessageObservable{AbstractMessage}}
    output_messages::Vector{MessageObservable{Message}}
    marginal::MarginalObservable
end

function randomvar()
    return RandomVariable(Vector{MessageObservable{AbstractMessage}}(), Vector{MessageObservable{Message}}(), MarginalObservable())
end

degree(randomvar::RandomVariable) = length(randomvar.input_messages)

israndom(::RandomVariable) = true
israndom(::AbstractArray{<:RandomVariable}) = true
isdata(::RandomVariable) = false
isdata(::AbstractArray{<:RandomVariable}) = false
isconst(::RandomVariable) = false
isconst(::AbstractArray{<:RandomVariable}) = false

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

const DefaultMessageProdFn = messages_prod_fn(FoldLeftProdStrategy(), GenericProd(), UnspecifiedFormConstraint(), FormConstraintCheckLast())
const DefaultMarginalProdFn = marginal_prod_fn(FoldLeftProdStrategy(), GenericProd(), UnspecifiedFormConstraint(), FormConstraintCheckLast())

struct RandomVariableActivationOptions{S, F, M}
    scheduler::S
    message_prod_fn::F
    marginal_prod_fn::M
end

RandomVariableActivationOptions() = RandomVariableActivationOptions(AsapScheduler(), DefaultMessageProdFn, DefaultMarginalProdFn)

function activate!(randomvar::RandomVariable, options::RandomVariableActivationOptions)
    d = length(randomvar.input_messages)
    outputmsgs = randomvar.output_messages
    resize!(outputmsgs, d)

    @inbounds for i in 1:d
        outputmsgs[i] = MessageObservable(Message)
    end

    if length(randomvar.input_messages) > 1
        chain = EqualityChain(randomvar.input_messages, schedule_on(options.scheduler), options.message_prod_fn)
        initialize!(chain, outputmsgs)
    elseif length(randomvar.input_messages) == 1
        # If the number of input message is equal to `1`,
        # than the output message is not producing any value
        connect!(outputmsgs[1], never(Message))
    else
        throw(ArgumentError("Cannot activate a random variable with zero or less than one inbound messages."))
    end

    _setmarginal!(randomvar, _makemarginal(randomvar, options))

    return nothing
end

_getmarginal(randomvar::RandomVariable) = randomvar.marginal
_setmarginal!(randomvar::RandomVariable, observable) = connect!(_getmarginal(randomvar), observable)
_makemarginal(randomvar::RandomVariable, options::RandomVariableActivationOptions) = begin
    return collectLatest(AbstractMessage, Marginal, randomvar.input_messages, options.marginal_prod_fn, reset_vstatus)
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
