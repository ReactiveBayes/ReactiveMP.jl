export RandomVariable, RandomVariableActivationOptions

## Random variable implementation

"""
    RandomVariable

A random variable defines properties that are needed for reactive message passing.

- `input_messages`: a vector of input messages streams, this should be created by nodes that are connected to this random variable. The nodes should use `getmessagein!` function.
- `output_messages`: a vector of output messages streams, this should be set by the random variable itself. The random variable should use `activate!` function when all nodes are connected.
- `marginal`: a marginal stream, this should be set by the random variable itself. The random variable should use `activate!` function when all nodes are connected.
- `messages_prod_fn`: a function that accepts a collection of messages and defines how to compute a product of those. Used in the outbound message computation.
- `marginal_prod_fn`: a function that accepts a collection of messages and defines how to compute a product of those. Used in the marginal computation.
"""
struct RandomVariable{M, F} <: AbstractVariable
    input_messages   :: Vector{MessageObservable{AbstractMessage}}
    output_messages  :: Vector{MessageObservable{Message}}
    marginal         :: MarginalObservable
    message_prod_fn  :: M
    marginal_prod_fn :: F
end

const DefaultMessageProdFn = messages_prod_fn(FoldLeftProdStrategy(), GenericProd(), UnspecifiedFormConstraint(), FormConstraintCheckLast()),
const DefaultMarginalProdFn = marginal_prod_fn(FoldLeftProdStrategy(), GenericProd(), UnspecifiedFormConstraint(), FormConstraintCheckLast())

function RandomVariable()
    return RandomVariable(DefaultMessageProdFn, DefaultMarginalProdFn)
end

function RandomVariable(messages_prod_fn::M, marginal_prod_fn::F) where {M, F}
    return RandomVariable{M, F}(Vector{MessageObservable{AbstractMessage}}(), Vector{MessageObservable{Message}}(), MarginalObservable(), messages_prod_fn, marginal_prod_fn)
end

degree(randomvar::RandomVariable) = length(randomvar.input_messages)

israndom(::RandomVariable)                  = true
israndom(::AbstractArray{<:RandomVariable}) = true
isdata(::RandomVariable)                    = false
isdata(::AbstractArray{<:RandomVariable})   = false
isconst(::RandomVariable)                   = false
isconst(::AbstractArray{<:RandomVariable})  = false

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

struct RandomVariableActivationOptions{S}
    scheduler::S
end

RandomVariableActivationOptions() = RandomVariableActivationOptions(AsapScheduler())

# options here must implement at least `Rocket.getscheduler`
function activate!(randomvar::RandomVariable, options::RandomVariableActivationOptions)

    # `5` here is empirical observation, maybe we can come up with better heuristic?
    # in case if number of connections is large we use cache equality nodes chain structure 
    if length(randomvar.input_messages) > 5
        initialize_output_messages!(randomvar, EqualityChain(randomvar.input_messages, schedule_on(options.scheduler), randomvar.message_prod_fn))
    else
        initialize_output_messages!(randomvar)
    end

    _setmarginal!(randomvar, _makemarginal(randomvar))

    return nothing
end

# Generic fallback for variables with small number of connected nodes, somewhere <= 5
# We do not create equality chain in this cases, but simply do eager product
function initialize_output_messages!(randomvar::RandomVariable)
    d          = length(randomvar.input_messages)
    outputmsgs = randomvar.output_messages
    inputmsgs  = randomvar.input_messages
    prod_fn    = randomvar.message_prod_fn

    resize!(outputmsgs, d)

    @inbounds for i in 1:d
        outputmsgs[i] = MessageObservable(Message)
        outputmsg     = collectLatest(AbstractMessage, Message, skipindex(inputmsgs, i), prod_fn)
        connect!(outputmsgs[i], outputmsg)
    end

    return nothing
end

# Equality chain initialisation for variables with large number of connected nodes, somewhere > 5
# In this cases it is more efficient to create an equality chain structure, but it does allocate way more memory
function initialize_output_messages!(randomvar::RandomVariable, chain::EqualityChain)
    d          = length(randomvar.input_messages)
    outputmsgs = randomvar.output_messages

    resize!(outputmsgs, d)

    @inbounds for i in 1:d
        outputmsgs[i] = MessageObservable(Message)
    end

    initialize!(chain, outputmsgs)

    return nothing
end

_getmarginal(randomvar::RandomVariable)              = randomvar.marginal
_setmarginal!(randomvar::RandomVariable, observable) = connect!(_getmarginal(randomvar), observable)
_makemarginal(randomvar::RandomVariable)             = collectLatest(AbstractMessage, Marginal, randomvar.input_messages, randomvar.marginal_prod_fn)

# name(randomvar::RandomVariable)            = randomvar.name
# isanonymous(randomvar::RandomVariable)     = randomvar.anonymous
# proxy_variables(randomvar::RandomVariable) = randomvar.proxy_variables
# collection_type(randomvar::RandomVariable) = randomvar.collection_type
# equality_chain(randomvar::RandomVariable)  = randomvar.equality_chain
# prod_constraint(randomvar::RandomVariable) = randomvar.prod_constraint
# prod_strategy(randomvar::RandomVariable)   = randomvar.prod_strategy

# marginal_form_constraint(randomvar::RandomVariable)     = randomvar.marginal_form_constraint
# marginal_form_check_strategy(randomvar::RandomVariable) = _marginal_form_check_strategy(randomvar.marginal_form_check_strategy, randomvar)

# _marginal_form_check_strategy(::FormConstraintCheckPickDefault, randomvar::RandomVariable) = default_form_check_strategy(marginal_form_constraint(randomvar))
# _marginal_form_check_strategy(form_check_strategy, randomvar::RandomVariable)              = form_check_strategy

# messages_form_constraint(randomvar::RandomVariable)     = randomvar.messages_form_constraint
# messages_form_check_strategy(randomvar::RandomVariable) = _messages_form_check_strategy(randomvar.messages_form_check_strategy, randomvar)

# _messages_form_check_strategy(::FormConstraintCheckPickDefault, randomvar::RandomVariable) = default_form_check_strategy(messages_form_constraint(randomvar))
# _messages_form_check_strategy(form_check_strategy, randomvar::RandomVariable)              = form_check_strategy
