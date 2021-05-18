export randomvar, simplerandomvar
export FoldLeftProdStrategy, FoldRightProdStrategy, AllAtOnceProdStrategy

# Messages to Marginal product strategies

struct FoldLeftProdStrategy end
struct FoldRightProdStrategy end

# Fallbacks to FoldLeftProdStrategy in case if there is no suitable method
struct AllAtOnceProdStrategy end

strategy_fn(::FoldLeftProdStrategy, prod_constraint)  = foldl_reduce_to_marginal(prod_constraint)
strategy_fn(::FoldRightProdStrategy, prod_constraint) = foldr_reduce_to_marginal(prod_constraint)
strategy_fn(::AllAtOnceProdStrategy, prod_constraint) = all_reduce_to_marginal(prod_constraint)

## Random variable implementation

mutable struct RandomVariable{C, P, S} <: AbstractVariable
    name             :: Symbol
    inputmsgs        :: Vector{LazyObservable{AbstractMessage}}
    local_constraint :: C
    prod_constraint  :: P
    prod_strategy    :: S
    marginal         :: Union{Nothing, MarginalObservable}
    portal           :: AbstractPortal
end

function randomvar(name::Symbol; local_constraint = Marginalisation(), prod_constraint = ProdAnalytical(), prod_strategy = FoldLeftProdStrategy()) 
    return RandomVariable(name, Vector{LazyObservable{AbstractMessage}}(), local_constraint, prod_constraint, prod_strategy, nothing, EmptyPortal())
end

function randomvar(name::Symbol, dims::Tuple; local_constraint = Marginalisation(), prod_constraint = ProdAnalytical(), prod_strategy = FoldLeftProdStrategy())
    return randomvar(name, dims...; local_constraint = local_constraint, prod_constraint = prod_constraint, prod_strategy = prod_strategy)
end

function randomvar(name::Symbol, dims::Vararg{Int}; local_constraint = Marginalisation(), prod_constraint = ProdAnalytical(), prod_strategy = FoldLeftProdStrategy())
    vars = Array{RandomVariable}(undef, dims)
    for index in CartesianIndices(axes(vars))
        @inbounds vars[index] = randomvar(Symbol(name, :_, Symbol(join(index.I, :_))); local_constraint = local_constraint, prod_constraint = prod_constraint, prod_strategy = prod_strategy)
    end
    return vars
end

degree(randomvar::RandomVariable)           = length(randomvar.inputmsgs)
name(randomvar::RandomVariable)             = randomvar.name
local_constraint(randomvar::RandomVariable) = randomvar.local_constraint
prod_constraint(randomvar::RandomVariable)  = randomvar.prod_constraint
prod_strategy(randomvar::RandomVariable)    = randomvar.prod_strategy

getlastindex(randomvar::RandomVariable) = length(randomvar.inputmsgs) + 1

messagein(randomvar::RandomVariable, index::Int)  = @inbounds randomvar.inputmsgs[index]
messageout(randomvar::RandomVariable, index::Int) = collectLatest(AbstractMessage, Message, skipindex(randomvar.inputmsgs, index), reduce_messages)

inbound_portal(randomvar::RandomVariable)          = randomvar.portal
inbound_portal!(randomvar::RandomVariable, portal) = randomvar.portal = (randomvar.portal + portal)

_getmarginal(randomvar::RandomVariable)                                = randomvar.marginal
_setmarginal!(randomvar::RandomVariable, marginal::MarginalObservable) = randomvar.marginal = marginal
_makemarginal(randomvar::RandomVariable)                               = collectLatest(AbstractMessage, Marginal, randomvar.inputmsgs, strategy_fn(prod_strategy(randomvar), prod_constraint(randomvar)))

function setmessagein!(randomvar::RandomVariable, index::Int, messagein)
    if index === length(randomvar.inputmsgs) + 1
        push!(randomvar.inputmsgs, messagein)
    else
        throw("TODO error message: setmessagein! in ::RandomVariable")
    end
end