export randomvar, simplerandomvar
export FoldLeftProdStrategy, FoldRightProdStrategy, AllAtOnceProdStrategy

# Messages to Marginal product strategies

struct FoldLeftProdStrategy end
struct FoldRightProdStrategy end

# Fallbacks to FoldLeftProdStrategy in case if there is no suitable method
struct AllAtOnceProdStrategy end

strategy_fn(::FoldLeftProdStrategy)  = foldl_reduce_to_marginal
strategy_fn(::FoldRightProdStrategy) = foldr_reduce_to_marginal
strategy_fn(::AllAtOnceProdStrategy) = all_reduce_to_marginal

## Random variable props

mutable struct RandomVariableProps
    marginal :: Union{Nothing, MarginalObservable}
    portal   :: AbstractPortal 

    RandomVariableProps() = new(nothing, EmptyPortal())
end

## Random variable implementation

struct RandomVariable{C, S} <: AbstractVariable
    name          :: Symbol
    inputmsgs     :: Vector{LazyObservable{AbstractMessage}}
    props         :: RandomVariableProps
    constraint    :: C
    prod_strategy :: S
end

function randomvar(name::Symbol; constraint = Marginalisation(), prod_strategy = FoldLeftProdStrategy()) 
    return RandomVariable(name, Vector{LazyObservable{AbstractMessage}}(), RandomVariableProps(), constraint, prod_strategy)
end

function randomvar(name::Symbol, dims::Tuple; constraint = Marginalisation(), prod_strategy = FoldLeftProdStrategy())
    return randomvar(name, dims...; constraint = constraint, prod_strategy = prod_strategy)
end

function randomvar(name::Symbol, dims::Vararg{Int}; constraint = Marginalisation(), prod_strategy = FoldLeftProdStrategy())
    vars = Array{RandomVariable}(undef, dims)
    for index in CartesianIndices(axes(vars))
        @inbounds vars[index] = randomvar(Symbol(name, :_, Symbol(join(index.I, :_))); constraint = constraint, prod_strategy = prod_strategy)
    end
    return vars
end

degree(randomvar::RandomVariable)        = length(randomvar.inputmsgs)
name(randomvar::RandomVariable)          = randomvar.name
constraint(randomvar::RandomVariable)    = randomvar.constraint
prod_strategy(randomvar::RandomVariable) = randomvar.prod_strategy

getlastindex(randomvar::RandomVariable) = length(randomvar.inputmsgs) + 1

messagein(randomvar::RandomVariable, index::Int)  = @inbounds randomvar.inputmsgs[index]
messageout(randomvar::RandomVariable, index::Int) = collectLatest(AbstractMessage, Message, skipindex(randomvar.inputmsgs, index), reduce_messages)

inbound_portal(randomvar::RandomVariable)          = randomvar.props.portal
inbound_portal!(randomvar::RandomVariable, portal) = randomvar.props.portal = portal

_getmarginal(randomvar::RandomVariable)                                = randomvar.props.marginal
_setmarginal!(randomvar::RandomVariable, marginal::MarginalObservable) = randomvar.props.marginal = marginal
_makemarginal(randomvar::RandomVariable)                               = collectLatest(AbstractMessage, Marginal, randomvar.inputmsgs, strategy_fn(prod_strategy(randomvar)))

function setmessagein!(randomvar::RandomVariable, index::Int, messagein)
    if index === length(randomvar.inputmsgs) + 1
        push!(randomvar.inputmsgs, messagein)
    else
        throw("TODO error message: setmessagein! in ::RandomVariable")
    end
end