export RandomVariable, randomvar

import Base: show

## Random variable implementation

mutable struct RandomVariable <: AbstractVariable
    name                :: Symbol
    collection_type     :: AbstractVariableCollectionType
    input_messages      :: Vector{MessageObservable{AbstractMessage}} 
    output_messages     :: Vector{MessageObservable{Message}}
    output_initialised  :: Bool
    output_cache        :: Union{Nothing, EqualityChain}
    marginal            :: MarginalObservable
    pipeline            :: AbstractPipelineStage
    proxy_variables     
    prod_constraint     
    prod_strategy       
    form_constraint     
    form_check_strategy
end

Base.show(io::IO, randomvar::RandomVariable) = print(io, "RandomVariable(", indexed_name(randomvar), ")")

function randomvar(name::Symbol, collection_type::AbstractVariableCollectionType = VariableIndividual(); pipeline = EmptyPipelineStage(), proxy_variables = nothing, prod_constraint = ProdAnalytical(), prod_strategy = FoldLeftProdStrategy(), form_constraint = UnspecifiedFormConstraint(), form_check_strategy = FormConstraintCheckPickDefault()) 
    return RandomVariable(name, collection_type, Vector{MessageObservable{AbstractMessage}}(), Vector{MessageObservable{Message}}(), false, nothing, MarginalObservable(), pipeline, proxy_variables, prod_constraint, prod_strategy, form_constraint, form_check_strategy)
end

function randomvar(name::Symbol, dims::Tuple; pipeline = EmptyPipelineStage(), prod_constraint = ProdAnalytical(), prod_strategy = FoldLeftProdStrategy(), form_constraint = UnspecifiedFormConstraint(), form_check_strategy = FormConstraintCheckPickDefault())
    return randomvar(name, dims...; pipeline = pipeline, prod_constraint = prod_constraint, prod_strategy = prod_strategy, form_constraint = form_constraint, form_check_strategy = form_check_strategy)
end

function randomvar(name::Symbol, length::Int; pipeline = EmptyPipelineStage(), prod_constraint = ProdAnalytical(), prod_strategy = FoldLeftProdStrategy(), form_constraint = UnspecifiedFormConstraint(), form_check_strategy = FormConstraintCheckPickDefault())
    vars = Vector{RandomVariable}(undef, length)
    @inbounds for i in 1:length
        vars[i] = randomvar(name, VariableVector(i); pipeline = pipeline, prod_constraint = prod_constraint, prod_strategy = prod_strategy, form_constraint = form_constraint, form_check_strategy = form_check_strategy)
    end
    return vars
end

function randomvar(name::Symbol, dims::Vararg{Int}; pipeline = EmptyPipelineStage(), prod_constraint = ProdAnalytical(), prod_strategy = FoldLeftProdStrategy(), form_constraint = UnspecifiedFormConstraint(), form_check_strategy = FormConstraintCheckPickDefault())
    vars = Array{RandomVariable}(undef, dims)
    size = axes(vars)
    @inbounds for i in CartesianIndices(axes(vars))
        vars[i] = randomvar(name, VariableArray(size, i); pipeline = pipeline, prod_constraint = prod_constraint, prod_strategy = prod_strategy, form_constraint = form_constraint, form_check_strategy = form_check_strategy)
    end
    return vars
end

degree(randomvar::RandomVariable)              = length(randomvar.input_messages)
name(randomvar::RandomVariable)                = randomvar.name
proxy(randomvar::RandomVariable)               = randomvar.proxy_variables
collection_type(randomvar::RandomVariable)     = randomvar.collection_type
equality_chain(randomvar::RandomVariable)      = randomvar.equality_chain
prod_constraint(randomvar::RandomVariable)     = randomvar.prod_constraint
prod_strategy(randomvar::RandomVariable)       = randomvar.prod_strategy
form_constraint(randomvar::RandomVariable)     = randomvar.form_constraint
form_check_strategy(randomvar::RandomVariable) = _form_check_strategy(randomvar.form_check_strategy, randomvar)

isproxy(randomvar::RandomVariable) = proxy(randomvar) !== nothing

israndom(::RandomVariable)                      = true
israndom(::AbstractVector{ <: RandomVariable }) = true

_form_check_strategy(::FormConstraintCheckPickDefault, randomvar::RandomVariable) = default_form_check_strategy(form_constraint(randomvar))
_form_check_strategy(form_check_strategy, randomvar::RandomVariable)              = form_check_strategy

messages_prod_fn(randomvar::RandomVariable) = messages_prod_fn(prod_strategy(randomvar), prod_constraint(randomvar), UnspecifiedFormConstraint(), default_form_check_strategy(UnspecifiedFormConstraint()))
marginal_prod_fn(randomvar::RandomVariable) = marginal_prod_fn(prod_strategy(randomvar), prod_constraint(randomvar), form_constraint(randomvar), form_check_strategy(randomvar))

getlastindex(randomvar::RandomVariable) = degree(randomvar) + 1

messagein(randomvar::RandomVariable, index::Int)  = @inbounds randomvar.input_messages[index]

function messageout(randomvar::RandomVariable, index::Int) 
    if randomvar.output_initialised === false
        initialize_output_messages!(randomvar)
    end
    return @inbounds randomvar.output_messages[index]
end

get_pipeline_stages(randomvar::RandomVariable)        = randomvar.pipeline
add_pipeline_stage!(randomvar::RandomVariable, stage) = randomvar.pipeline = (randomvar.pipeline + stage)

_getmarginal(randomvar::RandomVariable)              = randomvar.marginal
_setmarginal!(randomvar::RandomVariable, observable) = connect!(_getmarginal(randomvar), observable)
_makemarginal(randomvar::RandomVariable)             = collectLatest(AbstractMessage, Marginal, randomvar.input_messages, marginal_prod_fn(randomvar))

function setmessagein!(randomvar::RandomVariable, index::Int, messagein)
    if index === degree(randomvar) + 1
        push!(randomvar.input_messages, messagein)
    else
        error("Inconsistent state in setmessagein! function for random variable $(randomvar). `index` should be equal to `degree(randomvar) + 1 = $(degree(randomvar) + 1)`, $(index) is given instead")
    end
end

function activate!(model, randomvar::RandomVariable)
    # `5` here is empirical observation, maybe we can come up with better heuristic?
    # in case if number of connections is large we use cache equality nodes chain structure 
    if degree(randomvar) > 5
        chain_pipeline  = schedule_on(global_reactive_scheduler(getoptions(model)))
        chain_prod_fn   = messages_prod_fn(randomvar)
        randomvar.output_cache = EqualityChain(randomvar.input_messages, chain_pipeline, chain_prod_fn)
    end

    _setmarginal!(randomvar, _makemarginal(randomvar))

    return nothing
end

initialize_output_messages!(randomvar::RandomVariable) = initialize_output_messages!(randomvar.output_cache, randomvar)

# Generic fallback for variables with small number of connected nodes, somewhere <= 5
# We do not create equality chain in this cases, but simply do eager product
function initialize_output_messages!(::Nothing, randomvar::RandomVariable)
    d          = degree(randomvar)
    outputmsgs = randomvar.output_messages
    inputmsgs  = randomvar.input_messages
    prod_fn    = messages_prod_fn(randomvar)

    resize!(outputmsgs, d)

    @inbounds for i in 1:d
        outputmsgs[i] = MessageObservable(Message)
        outputmsg     = collectLatest(AbstractMessage, Message, skipindex(inputmsgs, i), prod_fn)
        connect!(outputmsgs[i], outputmsg)
    end

    randomvar.output_initialised = true

    return nothing
end

# Equality chain initialisation for variables with large number of connected nodes, somewhere > 5
# In this cases it is more efficient to create an equality chain structure, but it does allocate way more memory
function initialize_output_messages!(chain::EqualityChain, randomvar::RandomVariable)
    d          = degree(randomvar)
    outputmsgs = randomvar.output_messages

    resize!(outputmsgs, d)

    @inbounds for i in 1:d
        outputmsgs[i] = MessageObservable(Message)
    end

    initialize!(chain, outputmsgs)
    randomvar.output_initialised = true

    return nothing
end