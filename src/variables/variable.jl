export AbstractVariable, degree
export is_clamped, is_marginalisation, is_moment_matching
export FoldLeftProdStrategy, FoldRightProdStrategy, CustomProdStrategy
export getprediction, getpredictions, getmarginal, getmarginals, setmarginal!, setmarginals!, name, as_variable
export setmessage!, setmessages!

using Rocket

abstract type AbstractVariable end

## Base interface extensions

Base.broadcastable(v::AbstractVariable) = Ref(v)

## Messages to Marginal product strategies

struct FoldLeftProdStrategy end
struct FoldRightProdStrategy end

struct CustomProdStrategy{F}
    prod_callback_generator::F
end

"""
    messages_prod_fn(strategy, prod_constraint, form_constraint, form_check_strategy)

Returns a suitable prod computation function for a given strategy and constraints

See also: [`FoldLeftProdStrategy`](@ref), [`FoldRightProdStrategy`](@ref), [`CustomProdStrategy`](@ref)
"""
function messages_prod_fn end

messages_prod_fn(::FoldLeftProdStrategy, prod_constraint, form_constraint, form_check_strategy)       = prod_foldl_reduce(prod_constraint, form_constraint, form_check_strategy)
messages_prod_fn(::FoldRightProdStrategy, prod_constraint, form_constraint, form_check_strategy)      = prod_foldr_reduce(prod_constraint, form_constraint, form_check_strategy)
messages_prod_fn(strategy::CustomProdStrategy, prod_constraint, form_constraint, form_check_strategy) = strategy.prod_callback_generator(prod_constraint, form_constraint, form_check_strategy)

function marginal_prod_fn(strategy, prod_constraint, form_constraint, form_check_strategy)
    return let prod_fn = messages_prod_fn(strategy, prod_constraint, form_constraint, form_check_strategy)
        return (messages) -> as_marginal(prod_fn(messages))
    end
end

# Helper functions

israndom(v::AbstractArray{<:AbstractVariable}) = all(israndom, v)
isdata(v::AbstractArray{<:AbstractVariable})   = all(isdata, v)
isconst(v::AbstractArray{<:AbstractVariable})  = all(isconst, v)

# Getters

getprediction(variable::AbstractVariable)                    = _getprediction(variable)
getpredictions(variables::AbstractArray{<:AbstractVariable}) = collectLatest(map(v -> getprediction(v), variables))

getmarginal(variable::AbstractVariable)                                      = getmarginal(variable, SkipInitial())
getmarginal(variable::AbstractVariable, skip_strategy::MarginalSkipStrategy) = apply_skip_filter(_getmarginal(variable), skip_strategy)

getmarginals(variables::AbstractArray{<:AbstractVariable})                                      = getmarginals(variables, SkipInitial())
getmarginals(variables::AbstractArray{<:AbstractVariable}, skip_strategy::MarginalSkipStrategy) = collectLatest(map(v -> getmarginal(v, skip_strategy), variables))

## Setters

### Marginals

setmarginal!(variable::AbstractVariable, marginal) = setmarginal!(getmarginal(variable, IncludeAll()), marginal)

setmarginals!(variables::AbstractArray{<:AbstractVariable}, marginal::PointMass)    = _setmarginals!(Base.HasLength(), variables, Iterators.repeated(marginal, length(variables)))
setmarginals!(variables::AbstractArray{<:AbstractVariable}, marginal::Distribution) = _setmarginals!(Base.HasLength(), variables, Iterators.repeated(marginal, length(variables)))
setmarginals!(variables::AbstractArray{<:AbstractVariable}, marginals)              = _setmarginals!(Base.IteratorSize(marginals), variables, marginals)

function _setmarginals!(::Base.IteratorSize, variables::AbstractArray{<:AbstractVariable}, marginals)
    @assert length(variables) == length(marginals) "Variables $(variables) and marginals $(marginals) should have the same length"
    foreach(zip(variables, marginals)) do (variable, marginal)
        setmarginal!(variable, marginal)
    end
end

function _setmarginals!(::Any, variables::AbstractArray{<:AbstractVariable}, marginals)
    error("setmarginals!() failed. Default value is neither an iterable object nor a distribution.")
end

### Messages

setmessage!(variable::AbstractVariable, index::Int, message) = setmessage!(messageout(variable, index), message)
setmessage!(variable::AbstractVariable, message)             = foreach(i -> setmessage!(variable, i, message), 1:degree(variable))

setmessages!(variables::AbstractArray{<:AbstractVariable}, message::PointMass)    = _setmessages!(Base.HasLength(), variables, Iterators.repeated(message, length(variables)))
setmessages!(variables::AbstractArray{<:AbstractVariable}, message::Distribution) = _setmessages!(Base.HasLength(), variables, Iterators.repeated(message, length(variables)))
setmessages!(variables::AbstractArray{<:AbstractVariable}, messages)              = _setmessages!(Base.IteratorSize(messages), variables, messages)

function _setmessages!(::Base.IteratorSize, variables::AbstractArray{<:AbstractVariable}, messages)
    @assert length(variables) == length(messages) "Variables $(variables) and messages $(messages) should have the same length"
    foreach(zip(variables, messages)) do (variable, message)
        setmessage!(variable, message)
    end
end

function _setmessages!(::Any, variables::AbstractArray{<:AbstractVariable}, marginals)
    error("setmessages!() failed. Default value is neither an iterable object nor a distribution.")
end
