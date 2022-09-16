export RandomProcess, RandomProcessCreationOptions, randomprocess

import Base: show

## Random variable implementation

mutable struct RandomProcess <: AbstractVariable
    name               :: Symbol
    test_input         :: AbstractArray
    train_input        :: AbstractArray
    anonymous          :: Bool
    collection_type    :: AbstractVariableCollectionType
    input_messages     :: Vector{MessageObservable{AbstractMessage}}
    output_messages    :: Vector{MessageObservable{Message}}
    output_initialised :: Bool
    output_cache       :: Union{Nothing, EqualityChain}
    marginal           :: MarginalObservable
    pipeline           :: AbstractPipelineStage
    proxy_variables
    prod_constraint
    prod_strategy
    marginal_form_constraint
    marginal_form_check_strategy
    messages_form_constraint
    messages_form_check_strategy
end

Base.show(io::IO, randomprocess::RandomProcess) = print(io, "RandomProcess(", indexed_name(randomprocess), ")")

struct RandomProcessCreationOptions{L, P, C, S, T, G, M, N}
    pipeline                     :: L
    proxy_variables              :: P
    prod_constraint              :: C
    prod_strategy                :: S
    marginal_form_constraint     :: T
    marginal_form_check_strategy :: G
    messages_form_constraint     :: M
    messages_form_check_strategy :: N
    # test_input                   :: X
    # train_input                  :: Y
end


prod_constraint(options::RandomProcessCreationOptions)              = options.prod_constraint
prod_strategy(options::RandomProcessCreationOptions)                = options.prod_strategy
marginal_form_constraint(options::RandomProcessCreationOptions)     = options.marginal_form_constraint
marginal_form_check_strategy(options::RandomProcessCreationOptions) = options.marginal_form_check_strategy
messages_form_constraint(options::RandomProcessCreationOptions)     = options.messages_form_constraint
messages_form_check_strategy(options::RandomProcessCreationOptions) = options.messages_form_check_strategy
# test_input(options::RandomProcessCreationOptions)                   = options.test_input
# train_input(options::RandomProcessCreationOptions)                   = options.train_input

function RandomProcessCreationOptions()
    # Return default settings
    return RandomProcessCreationOptions(
        EmptyPipelineStage(),             # pipeline
        nothing,                          # proxy_variables
        ProdAnalytical(),                 # prod_constraint
        FoldLeftProdStrategy(),           # prod_strategy
        UnspecifiedFormConstraint(),      # marginal_form_constraint
        FormConstraintCheckPickDefault(), # marginal_form_check_strategy
        UnspecifiedFormConstraint(),      # messages_form_constraint
        FormConstraintCheckPickDefault()  # messages_form_check_strategy
        # nothing,                          # test_input
        # nothing,                          # train_input
    )
end

const EmptyRandomProcessCreationOptions =
    RandomProcessCreationOptions(nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)

randomprocess_options_set_pipeline(pipeline)                                         = randomprocess_options_set_pipeline(RandomProcessCreationOptions(), pipeline)
randomprocess_options_set_proxy_variables(proxy_variables)                           = randomprocess_options_set_proxy_variables(RandomProcessCreationOptions(), proxy_variables)
randomprocess_options_set_prod_constraint(prod_constraint)                           = randomprocess_options_set_prod_constraint(RandomProcessCreationOptions(), prod_constraint)
randomprocess_options_set_prod_strategy(prod_strategy)                               = randomprocess_options_set_prod_strategy(RandomProcessCreationOptions(), prod_strategy)
randomprocess_options_set_marginal_form_constraint(marginal_form_constraint)         = randomprocess_options_set_marginal_form_constraint(RandomProcessCreationOptions(), marginal_form_constraint)
randomprocess_options_set_marginal_form_check_strategy(marginal_form_check_strategy) = randomprocess_options_set_marginal_form_check_strategy(RandomProcessCreationOptions(), marginal_form_check_strategy)
randomprocess_options_set_messages_form_constraint(messages_form_constraint)         = randomprocess_options_set_messages_form_constraint(RandomProcessCreationOptions(), messages_form_constraint)
randomprocess_options_set_messages_form_check_strategy(messages_form_check_strategy) = randomprocess_options_set_messages_form_check_strategy(RandomProcessCreationOptions(), messages_form_check_strategy)
# randomprocess_options_set_test_input(test_input)                                     = randomprocess_options_set_test_input(RandomProcessCreationOptions(),test_input)
# randomprocess_options_set_train_input(train_input)                                   = randomprocess_options_set_train_input(RandomProcessCreationOptions(),train_input)


randomprocess_options_set_pipeline(options::RandomProcessCreationOptions, pipeline)                                         = RandomProcessCreationOptions(pipeline, options.proxy_variables, options.prod_constraint, options.prod_strategy, options.marginal_form_constraint, options.marginal_form_check_strategy, options.messages_form_constraint, options.messages_form_check_strategy)
randomprocess_options_set_proxy_variables(options::RandomProcessCreationOptions, proxy_variables)                           = RandomProcessCreationOptions(options.pipeline, proxy_variables, options.prod_constraint, options.prod_strategy, options.marginal_form_constraint, options.marginal_form_check_strategy, options.messages_form_constraint, options.messages_form_check_strategy)
randomprocess_options_set_prod_constraint(options::RandomProcessCreationOptions, prod_constraint)                           = RandomProcessCreationOptions(options.pipeline, options.proxy_variables, prod_constraint, options.prod_strategy, options.marginal_form_constraint, options.marginal_form_check_strategy, options.messages_form_constraint, options.messages_form_check_strategy)
randomprocess_options_set_prod_strategy(options::RandomProcessCreationOptions, prod_strategy)                               = RandomProcessCreationOptions(options.pipeline, options.proxy_variables, options.prod_constraint, prod_strategy, options.marginal_form_constraint, options.marginal_form_check_strategy, options.messages_form_constraint, options.messages_form_check_strategy)
randomprocess_options_set_marginal_form_constraint(options::RandomProcessCreationOptions, marginal_form_constraint)         = RandomProcessCreationOptions(options.pipeline, options.proxy_variables, options.prod_constraint, options.prod_strategy, marginal_form_constraint, options.marginal_form_check_strategy, options.messages_form_constraint, options.messages_form_check_strategy)
randomprocess_options_set_marginal_form_check_strategy(options::RandomProcessCreationOptions, marginal_form_check_strategy) = RandomProcessCreationOptions(options.pipeline, options.proxy_variables, options.prod_constraint, options.prod_strategy, options.marginal_form_constraint, marginal_form_check_strategy, options.messages_form_constraint, options.messages_form_check_strategy)
randomprocess_options_set_messages_form_constraint(options::RandomProcessCreationOptions, messages_form_constraint)         = RandomProcessCreationOptions(options.pipeline, options.proxy_variables, options.prod_constraint, options.prod_strategy, options.marginal_form_constraint, options.marginal_form_check_strategy, messages_form_constraint, options.messages_form_check_strategy)
randomprocess_options_set_messages_form_check_strategy(options::RandomProcessCreationOptions, messages_form_check_strategy) = RandomProcessCreationOptions(options.pipeline, options.proxy_variables, options.prod_constraint, options.prod_strategy, options.marginal_form_constraint, options.marginal_form_check_strategy, options.messages_form_constraint, messages_form_check_strategy)
# randomprocess_options_set_test_input(options::RandomProcessCreationOptions,test_input)                                      = RandomProcessCreationOptions(options.pipeline, options.proxy_variables, options.prod_constraint, options.prod_strategy, options.marginal_form_constraint, options.marginal_form_check_strategy, options.messages_form_constraint, options.messages_form_check_strategy, test_input,options.train_input)
# randomprocess_options_set_train_input(options::RandomProcessCreationOptions,train_input)                                    = RandomProcessCreationOptions(options.pipeline, options.proxy_variables, options.prod_constraint, options.prod_strategy, options.marginal_form_constraint, options.marginal_form_check_strategy, options.messages_form_constraint, options.messages_form_check_strategy, options.test_input,train_input)

function randomprocess end

randomprocess(name::Symbol, test_input::AbstractArray, train_input::AbstractArray)                                                  = randomprocess(RandomProcessCreationOptions(), name,test_input,train_input, VariableIndividual())
randomprocess(name::Symbol, test_input::AbstractArray, train_input::AbstractArray, collection_type::AbstractVariableCollectionType) = randomprocess(RandomProcessCreationOptions(), name, test_input, train_input, collection_type)
randomprocess(name::Symbol, test_input::AbstractArray, train_input::AbstractArray, length::Int)                                     = randomprocess(RandomProcessCreationOptions(), name, test_input, train_input, length)
randomprocess(name::Symbol, test_input::AbstractArray, train_input::AbstractArray, dims::Tuple)                                     = randomprocess(RandomProcessCreationOptions(), name, test_input, train_input, dims)
randomprocess(name::Symbol, test_input::AbstractArray, train_input::AbstractArray, dims::Vararg{Int})                               = randomprocess(RandomProcessCreationOptions(), name, test_input, train_input, dims)

randomprocess(options::RandomProcessCreationOptions, name::Symbol, test_input::AbstractArray, train_input::AbstractArray)                    = randomprocess(options, name, test_input, train_input, VariableIndividual())
randomprocess(options::RandomProcessCreationOptions, name::Symbol, test_input::AbstractArray, train_input::AbstractArray, dims::Vararg{Int}) = randomprocess(options, name, test_input, train_input, dims)

function randomprocess(
    options::RandomProcessCreationOptions,
    name::Symbol,
    test_input::AbstractArray,
    train_input::AbstractArray,
    collection_type::AbstractVariableCollectionType
)
    return RandomProcess(
        name,
        test_input,
        train_input,
        false,
        collection_type,
        Vector{MessageObservable{AbstractMessage}}(),
        Vector{MessageObservable{Message}}(),
        false,
        nothing,
        MarginalObservable(),
        something(options.pipeline, EmptyPipelineStage()),      # `something(args..)` returns the first `notnothing` object
        options.proxy_variables,                                # here we do allow `nothing`, so no need for `something(...)`
        something(options.prod_constraint, ProdAnalytical()),
        something(options.prod_strategy, FoldLeftProdStrategy()),
        something(options.marginal_form_constraint, UnspecifiedFormConstraint()),
        something(options.marginal_form_check_strategy, FormConstraintCheckPickDefault()),
        something(options.messages_form_constraint, UnspecifiedFormConstraint()),
        something(options.messages_form_check_strategy, FormConstraintCheckPickDefault())
    )
end

degree(randomprocess::RandomProcess)          = length(randomprocess.input_messages)
name(randomprocess::RandomProcess)            = randomprocess.name
isanonymous(randomprocess::RandomProcess)     = randomprocess.anonymous
proxy_variables(randomprocess::RandomProcess) = randomprocess.proxy_variables
collection_type(randomprocess::RandomProcess) = randomprocess.collection_type
equality_chain(randomprocess::RandomProcess)  = randomprocess.equality_chain
prod_constraint(randomprocess::RandomProcess) = randomprocess.prod_constraint
prod_strategy(randomprocess::RandomProcess)   = randomprocess.prod_strategy
test_input(randomprocess::RandomProcess)      = randomprocess.test_input
train_input(randomprocess::RandomProcess)     = randomprocess.train_input

marginal_form_constraint(randomprocess::RandomProcess)     = randomprocess.marginal_form_constraint
marginal_form_check_strategy(randomprocess::RandomProcess) = _marginal_form_check_strategy(randomprocess.marginal_form_check_strategy, randomprocess)

_marginal_form_check_strategy(::FormConstraintCheckPickDefault, randomprocess::RandomProcess) = default_form_check_strategy(marginal_form_constraint(randomprocess))
_marginal_form_check_strategy(form_check_strategy, randomprocess::RandomProcess)              = form_check_strategy

messages_form_constraint(randomprocess::RandomProcess)     = randomprocess.messages_form_constraint
messages_form_check_strategy(randomprocess::RandomProcess) = _messages_form_check_strategy(randomprocess.messages_form_check_strategy, randomprocess)

_messages_form_check_strategy(::FormConstraintCheckPickDefault, randomprocess::RandomProcess) = default_form_check_strategy(messages_form_constraint(randomprocess))
_messages_form_check_strategy(form_check_strategy, randomprocess::RandomProcess)              = form_check_strategy

isproxy(randomprocess::RandomProcess) = proxy_variables(randomprocess) !== nothing

isprocess(::RandomProcess)                 = true
isprocess(::AbstractArray{<:RandomProcess})= true
israndom(::RandomProcess)                  = true
israndom(::AbstractArray{<:RandomProcess}) = true
isdata(::RandomProcess)                    = false
isdata(::AbstractArray{<:RandomProcess})   = false
isconst(::RandomProcess)                   = false
isconst(::AbstractArray{<:RandomProcess})  = false


# messages_prod_fn(randomprocess::RandomProcess) = messages_prod_fn(
#     prod_strategy(randomprocess),
#     prod_constraint(randomprocess),
#     messages_form_constraint(randomprocess),
#     messages_form_check_strategy(randomprocess),
#     test_input(randomprocess),
#     train_input(randomprocess)
# )
# marginal_prod_fn(randomprocess::RandomProcess) = marginal_prod_fn(
#     prod_strategy(randomprocess),
#     prod_constraint(randomprocess),
#     marginal_form_constraint(randomprocess),
#     marginal_form_check_strategy(randomprocess),
#     test_input(randomprocess),
#     train_input(randomprocess)
# )

getlastindex(randomprocess::RandomProcess) = degree(randomprocess) + 1

messagein(randomprocess::RandomProcess, index::Int) = @inbounds randomprocess.input_messages[index]

function messageout(randomprocess::RandomProcess, index::Int)
    if randomprocess.output_initialised === false
        initialize_output_messages!(randomprocess)
    end
    return @inbounds randomprocess.output_messages[index]
end


get_pipeline_stages(randomprocess::RandomProcess)        = randomprocess.pipeline
add_pipeline_stage!(randomprocess::RandomProcess, stage) = randomprocess.pipeline = (randomprocess.pipeline + stage)

_getmarginal(randomprocess::RandomProcess)              = randomprocess.marginal
_setmarginal!(randomprocess::RandomProcess, observable) = connect!(_getmarginal(randomprocess), observable)
_makemarginal(randomprocess::RandomProcess)             = collectLatest(AbstractMessage, Marginal, randomprocess.input_messages, marginal_prod_fn(randomprocess))

setanonymous!(randomprocess::RandomProcess, anonymous::Bool) = randomprocess.anonymous = anonymous

function setmessagein!(randomprocess::RandomProcess, index::Int, messagein)
    if index === degree(randomprocess) + 1
        push!(randomprocess.input_messages, messagein)
    else
        error(
            "Inconsistent state in setmessagein! function for random variable $(randomprocess). `index` should be equal to `degree(randomprocess) + 1 = $(degree(randomprocess) + 1)`, $(index) is given instead"
        )
    end
end

function activate!(model, randomprocess::RandomProcess)
    if randomprocess.output_initialised === true
        error("Broken random variable ", randomprocess, ". Unreachable reached.")
    end

    # `5` here is empirical observation, maybe we can come up with better heuristic?
    # in case if number of connections is large we use cache equality nodes chain structure 
    if degree(randomprocess) > 1000
        chain_pipeline = schedule_on(global_reactive_scheduler(getoptions(model)))
        chain_prod_fn = messages_prod_fn(randomprocess)
        randomprocess.output_cache = EqualityChain(randomprocess.input_messages, chain_pipeline, chain_prod_fn)
    end

    _setmarginal!(randomprocess, _makemarginal(randomprocess))

    return nothing
end

initialize_output_messages!(randomprocess::RandomProcess) = initialize_output_messages!(randomprocess.output_cache, randomprocess)

# Generic fallback for variables with small number of connected nodes, somewhere <= 5
# We do not create equality chain in this cases, but simply do eager product
function initialize_output_messages!(::Nothing, randomprocess::RandomProcess)
    d          = degree(randomprocess)
    outputmsgs = randomprocess.output_messages
    inputmsgs  = randomprocess.input_messages
    prod_fn    = messages_prod_fn(randomprocess)
    resize!(outputmsgs, d)

    @inbounds for i in 1:d
        outputmsgs[i] = MessageObservable(Message)
        outputmsg     = collectLatest(AbstractMessage, Message, skipindex(inputmsgs, i), prod_fn)
        connect!(outputmsgs[i], outputmsg)
    end

    
    randomprocess.output_initialised = true

    return nothing
end

# Equality chain initialisation for variables with large number of connected nodes, somewhere > 5
# In this cases it is more efficient to create an equality chain structure, but it does allocate way more memory
function initialize_output_messages!(chain::EqualityChain, randomprocess::RandomProcess)
    d          = degree(randomprocess)
    outputmsgs = randomprocess.output_messages

    resize!(outputmsgs, d)

    @inbounds for i in 1:d
        outputmsgs[i] = MessageObservable(Message)
    end

    initialize!(chain, outputmsgs)
    randomprocess.output_initialised = true

    return nothing
end