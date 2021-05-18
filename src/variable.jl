export AbstractVariable, degree
export AbstractVariableLocalConstraint, ClampedVariable, Marginalisation
export is_clamped, is_marginalisation, is_moment_matching
export getmarginal, getmarginals, setmarginal!, setmarginals!, name, as_variable

using Rocket

abstract type AbstractVariable end

## Variable constraints

abstract type AbstractVariableLocalConstraint end

struct ClampedVariable <: AbstractVariableLocalConstraint end
struct Marginalisation <: AbstractVariableLocalConstraint end
struct MomentMatching  <: AbstractVariableLocalConstraint end # TODO: WIP

is_clamped(variable::AbstractVariable)        = is_clamped(local_constraint(variable))
is_clamped(::AbstractVariableLocalConstraint) = false
is_clamped(::ClampedVariable)                 = true

is_marginalisation(variable::AbstractVariable)        = is_marginalisation(local_constraint(variable))
is_marginalisation(::AbstractVariableLocalConstraint) = false
is_marginalisation(::Marginalisation)                 = true

is_moment_matching(variable::AbstractVariable)        = is_moment_matching(local_constraint(variable))
is_moment_matching(::AbstractVariableLocalConstraint) = false
is_moment_matching(::MomentMatching)                  = true

## Messages to Marginal product strategies

struct FoldLeftProdStrategy end
struct FoldRightProdStrategy end

struct CustomProdStrategy{F}
    prod_callback_generator :: F
end

"""
    prod_fn(strategy, constraint)

Returns a suitable prod computation function for a given strategy and constraint

See also: [`FoldLeftProdStrategy`](@ref), [`FoldRightProdStrategy`](@ref), [`CustomProdStrategy`](@ref)
"""
function prod_fn end

prod_fn(::FoldLeftProdStrategy, prod_constraint)       = foldl_reduce_to_marginal(prod_constraint)
prod_fn(::FoldRightProdStrategy, prod_constraint)      = foldr_reduce_to_marginal(prod_constraint)
prod_fn(strategy::CustomProdStrategy, prod_constraint) = strategy.prod_callback_generator(prod_constraint)

## Common functions

function degree end

inbound_portal!(variable::AbstractVariable, portal) = error("Its not possible to change an inbound portal for $(variable)")

# Helper functions
# Getters

getmarginal(variable::AbstractVariable) = getmarginal(variable, SkipInitial())

function getmarginal(variable::AbstractVariable, skip_strategy::MarginalSkipStrategy)
    vmarginal = _getmarginal(variable)
    if vmarginal === nothing
        vmarginal = as_marginal_observable(_makemarginal(variable))
        _setmarginal!(variable, vmarginal)
    end
    return as_marginal_observable(vmarginal, skip_strategy)
end

getmarginals(variables::AbstractArray{ <: AbstractVariable })                                      = getmarginals(variables, SkipInitial())
getmarginals(variables::AbstractArray{ <: AbstractVariable }, skip_strategy::MarginalSkipStrategy) = collectLatest(map(v -> getmarginal(v, skip_strategy), variables))

# Setters

function setmarginal!(variable::AbstractVariable, marginal)
    setmarginal!(getmarginal(variable, IncludeAll()), marginal)
end

setmarginals!(variables::AbstractArray{ <: AbstractVariable }, marginal::Distribution)    = _setmarginals!(Base.HasLength(), variables, Iterators.repeated(marginal, length(variables)))
setmarginals!(variables::AbstractArray{ <: AbstractVariable }, marginals)                 = _setmarginals!(Base.IteratorSize(marginals), variables, marginals)

function _setmarginals!(::Base.IteratorSize, variables::AbstractArray{ <: AbstractVariable }, marginals)
    foreach(zip(variables, marginals)) do (variable, marginal)
        setmarginal!(getmarginal(variable, IncludeAll()), marginal)
    end
end

function _setmarginals!(::Any, variables::AbstractArray{ <: AbstractVariable }, marginals)
    error("setmarginals!() failed. Default value is neither an iterable object nor a distribution.")
end

function name(variable::AbstractVariable)
    return variable.name
end

## Helper functions

as_variable(x)                   = constvar(gensym(:as_var), x)
as_variable(v::AbstractVariable) = v
as_variable(t::Tuple)            = map(as_variable, t)


