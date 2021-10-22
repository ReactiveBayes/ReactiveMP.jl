export RandomVariable, randomvar

## Random variable implementation

mutable struct RandomVariable <: AbstractVariable
    name                :: Symbol
    inputmsgs           :: Vector{MessageObservable{AbstractMessage}} 
    outputmsgs          :: Vector{MessageObservable{Message}}
    marginal            :: Union{Nothing, MarginalObservable}
    equality_chain      :: Union{Nothing, EqualityChain}
    pipeline            :: AbstractPipelineStage
    prod_constraint     
    prod_strategy       
    form_constraint     
    form_check_strategy
    
end

function randomvar(name::Symbol; pipeline = EmptyPipelineStage(), prod_constraint = ProdAnalytical(), prod_strategy = FoldLeftProdStrategy(), form_constraint = UnspecifiedFormConstraint(), form_check_strategy = FormConstraintCheckPickDefault()) 
    return RandomVariable(name, Vector{MessageObservable{AbstractMessage}}(), Vector{MessageObservable{Message}}(), nothing, nothing, pipeline, prod_constraint, prod_strategy, form_constraint, form_check_strategy)
end

function randomvar(name::Symbol, dims::Tuple; pipeline = EmptyPipelineStage(), prod_constraint = ProdAnalytical(), prod_strategy = FoldLeftProdStrategy(), form_constraint = UnspecifiedFormConstraint(), form_check_strategy = FormConstraintCheckPickDefault())
    return randomvar(name, dims...; pipeline = pipeline, prod_constraint = prod_constraint, prod_strategy = prod_strategy, form_constraint = form_constraint, form_check_strategy = form_check_strategy)
end

function randomvar(name::Symbol, dims::Vararg{Int}; pipeline = EmptyPipelineStage(), prod_constraint = ProdAnalytical(), prod_strategy = FoldLeftProdStrategy(), form_constraint = UnspecifiedFormConstraint(), form_check_strategy = FormConstraintCheckPickDefault())
    vars = Array{RandomVariable}(undef, dims)
    for i in CartesianIndices(axes(vars))
        @inbounds vars[i] = randomvar(Symbol(name, :_, Symbol(join(i.I, :_))); pipeline = pipeline, prod_constraint = prod_constraint, prod_strategy = prod_strategy, form_constraint = form_constraint, form_check_strategy = form_check_strategy)
    end
    return vars
end

degree(randomvar::RandomVariable)              = length(randomvar.inputmsgs)
name(randomvar::RandomVariable)                = randomvar.name
equality_chain(randomvar::RandomVariable)      = randomvar.equality_chain
prod_constraint(randomvar::RandomVariable)     = randomvar.prod_constraint
prod_strategy(randomvar::RandomVariable)       = randomvar.prod_strategy
form_constraint(randomvar::RandomVariable)     = randomvar.form_constraint
form_check_strategy(randomvar::RandomVariable) = _form_check_strategy(randomvar.form_check_strategy, randomvar)

_form_check_strategy(::FormConstraintCheckPickDefault, randomvar::RandomVariable) = default_form_check_strategy(form_constraint(randomvar))
_form_check_strategy(form_check_strategy, randomvar::RandomVariable)              = form_check_strategy

messages_prod_fn(randomvar::RandomVariable) = messages_prod_fn(prod_strategy(randomvar), prod_constraint(randomvar), UnspecifiedFormConstraint(), default_form_check_strategy(UnspecifiedFormConstraint()))
marginal_prod_fn(randomvar::RandomVariable) = marginal_prod_fn(prod_strategy(randomvar), prod_constraint(randomvar), form_constraint(randomvar), form_check_strategy(randomvar))

getlastindex(randomvar::RandomVariable) = degree(randomvar) + 1

messagein(randomvar::RandomVariable, index::Int)  = @inbounds randomvar.inputmsgs[index]
messageout(randomvar::RandomVariable, index::Int) = @inbounds randomvar.outputmsgs[index]

get_pipeline_stages(randomvar::RandomVariable)        = randomvar.pipeline
add_pipeline_stage!(randomvar::RandomVariable, stage) = randomvar.pipeline = (randomvar.pipeline + stage)

_getmarginal(randomvar::RandomVariable)                                = randomvar.marginal
_setmarginal!(randomvar::RandomVariable, marginal::MarginalObservable) = randomvar.marginal = marginal
_makemarginal(randomvar::RandomVariable)                               = collectLatest(AbstractMessage, Marginal, randomvar.inputmsgs, marginal_prod_fn(randomvar))

function setmessagein!(randomvar::RandomVariable, index::Int, messagein)
    if index === degree(randomvar) + 1
        push!(randomvar.inputmsgs, messagein)
    else
        error("Inconsistent state in setmessagein! function for random variable $(randomvar). `index` should be equal to `degree(randomvar) + 1 = $(degree(randomvar) + 1)`, $(index) is given instead")
    end
end

function activate!(model, randomvar::RandomVariable)
    d = degree(randomvar)
    resize!(randomvar.outputmsgs, d)
    # `5` here is empirical observation, maybe we can come up with better heuristic?
    if d > 5
        randomvar.equality_chain = EqualityChain(d, randomvar.inputmsgs, messages_prod_fn(randomvar))
        activate!(model, randomvar.equality_chain, randomvar.inputmsgs, randomvar.outputmsgs)
    else
        for index in 1:d
            messageout = collectLatest(AbstractMessage, Message, skipindex(randomvar.inputmsgs, index), messages_prod_fn(randomvar))
            @inbounds randomvar.outputmsgs[index] = as_message_observable(messageout)
        end        
    end
    return nothing
end