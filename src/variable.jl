export AbstractVariable, degree
export is_clamped, is_marginalisation, is_moment_matching
export FoldLeftProdStrategy, FoldRightProdStrategy, CustomProdStrategy
export getmarginal, getmarginals, setmarginal!, setmarginals!, name, as_variable
export setmessage!, setmessages!

using Rocket

abstract type AbstractVariable end

## Variable collection type

abstract type AbstractVariableCollectionType end

struct VariableIndividual <: AbstractVariableCollectionType end

struct VariableVector <: AbstractVariableCollectionType
    index :: Int
end

struct VariableArray{S, I} <: AbstractVariableCollectionType
    size  :: S
    index :: I
end

linear_index(::VariableIndividual) = nothing
linear_index(v::VariableVector)    = v.index
linear_index(v::VariableArray)     = LinearIndices(v.size)[v.index]

indexed_name(::VariableIndividual, name::Symbol) = string(name)
indexed_name(seq::VariableVector, name::Symbol)  = string(name, "_", seq.index)
indexed_name(array::VariableArray, name::Symbol) = string(name, "_", join(array.index.I, "_"))

indexed_name(randomvar::AbstractVariable) = indexed_name(collection_type(randomvar), name(randomvar))

## Messages to Marginal product strategies

struct FoldLeftProdStrategy end
struct FoldRightProdStrategy end

struct CustomProdStrategy{F}
    prod_callback_generator :: F
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

## Common functions

function degree end

add_pipeline_stage!(variable::AbstractVariable, stage) = error("Its not possible to add a new pipeline stage for $(variable)")

# Helper functions
# Getters

getmarginal(variable::AbstractVariable)                                      = getmarginal(variable, SkipInitial())
getmarginal(variable::AbstractVariable, skip_strategy::MarginalSkipStrategy) = apply_skip_filter(_getmarginal(variable), skip_strategy)

getmarginals(variables::AbstractArray{ <: AbstractVariable })                                      = getmarginals(variables, SkipInitial())
getmarginals(variables::AbstractArray{ <: AbstractVariable }, skip_strategy::MarginalSkipStrategy) = collectLatest(map(v -> getmarginal(v, skip_strategy), variables))

## Setters

### Marginals

setmarginal!(variable::AbstractVariable, marginal) = setmarginal!(getmarginal(variable, IncludeAll()), marginal)

setmarginals!(variables::AbstractArray{ <: AbstractVariable }, marginal::Distribution)    = _setmarginals!(Base.HasLength(), variables, Iterators.repeated(marginal, length(variables)))
setmarginals!(variables::AbstractArray{ <: AbstractVariable }, marginals)                 = _setmarginals!(Base.IteratorSize(marginals), variables, marginals)

function _setmarginals!(::Base.IteratorSize, variables::AbstractArray{ <: AbstractVariable }, marginals)
    foreach(zip(variables, marginals)) do (variable, marginal)
        setmarginal!(variable, marginal)
    end
end

function _setmarginals!(::Any, variables::AbstractArray{ <: AbstractVariable }, marginals)
    error("setmarginals!() failed. Default value is neither an iterable object nor a distribution.")
end

### Messages

setmessage!(variable::AbstractVariable, index::Int, message) = setmessage!(messageout(variable, index), message)
setmessage!(variable::AbstractVariable, message)             = foreach(i -> setmessage!(variable, i, message), 1:degree(variable))

setmessages!(variables::AbstractArray{ <: AbstractVariable }, message::Distribution)    = _setmessages!(Base.HasLength(), variables, Iterators.repeated(message, length(variables)))
setmessages!(variables::AbstractArray{ <: AbstractVariable }, messages)                 = _setmessages!(Base.IteratorSize(messages), variables, messages)

function _setmessages!(::Base.IteratorSize, variables::AbstractArray{ <: AbstractVariable }, messages)
    foreach(zip(variables, messages)) do (variable, message)
        setmessage!(variable, message)
    end
end

function _setmessages!(::Any, variables::AbstractArray{ <: AbstractVariable }, marginals)
    error("setmessages!() failed. Default value is neither an iterable object nor a distribution.")
end

##

function name(variable::AbstractVariable)
    return variable.name
end

##

struct VariableReferenceProxyUnchecked end
struct VariableReferenceProxyChecked end

"""
    resolve_variable_proxy

This function resolves variable that should be used for factorisation constraints resolution or in other places. The idea here is that random variables can be automatically created by the model specification 
language and user might be unaware of them. Such variables also have some randomly generated names and cannot be used explicitly in constraints specification language. However, ReactiveMP.jl keeps 
track of `proxy_variables`. During the first call of `get_factorisation_reference` we check if there are some proxy variables at all and:
1. if not we simply return name and linear index of the current variable
2. if yes we pass it futher to the `unchecked` version of the function 
   2.1 `unchecked` version return immediatelly if there is only one proxy var (see bullet 1)
   2.2 in case of multiple proxy vars we filter only `RandomVariable` and call `checked` version of the function 
3. `checked` version of the function return immediatelly if there is only one proxy random variable left, if there are multuple proxy random vars we throw an error as this case is ambigous for factorisation constrains specification

This function is a part of private API and should not be used explicitly.
"""
function resolve_variable_proxy end

resolve_variable_proxy(var::AbstractVariable) = resolve_variable_proxy(var, VariableReferenceProxyUnchecked(), proxy(var))

resolve_variable_proxy(var::AbstractVariable, ::Union{ VariableReferenceProxyChecked, VariableReferenceProxyUnchecked }, ::Nothing) = (name(var), linear_index(collection_type(var)), var)

resolve_variable_proxy(var::AbstractVariable, ::VariableReferenceProxyUnchecked, proxy::Tuple{T}) where { T <: AbstractVariable } = resolve_variable_proxy(first(proxy))
resolve_variable_proxy(var::AbstractVariable, ::VariableReferenceProxyUnchecked, proxy::Tuple)                                    = resolve_variable_proxy(var, VariableReferenceProxyChecked(), filter(v -> v isa RandomVariable, proxy))

resolve_variable_proxy(::AbstractVariable, ::VariableReferenceProxyChecked, proxy::Tuple{T})   where { T <: AbstractVariable } = resolve_variable_proxy(first(proxy))
resolve_variable_proxy(::AbstractVariable, ::VariableReferenceProxyChecked, proxy::Tuple)                                      = error("Multiple proxy vars in variable reference resolution are dissalowed. This may happened because of the deterministic relation in the model that has more than one input. This setting does not play nicely with constraints or meta specification languages. As a workaround create and give a specific name for the output variable of this deterministic relation.")

## Helper functions

as_variable(x)                   = constvar(gensym(:as_var), x)
as_variable(v::AbstractVariable) = v
as_variable(t::Tuple)            = map(as_variable, t)

israndom(v::AbstractVector{ <: AbstractVariable }) = all(israndom, v)
isproxy(v::AbstractVector{ <: AbstractVariable })  = any(isproxy, v)

