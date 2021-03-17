export AbstractVariable, degree
export ClampedVariable, Marginalisation, ExpectationMaximisation, EM
export RandomVariable, randomvar
export SimpleRandomVariable, simplerandomvar
export ConstVariable, constvar
export DataVariable, datavar, update!, finish!
export getmarginal, getmarginals, setmarginal!, setmarginals!, activate!, name
export as_variable

using Rocket

abstract type AbstractVariable end

## Variable constraints

struct ClampedVariable end
struct Marginalisation end
struct ExpectationMaximisation end

const EM = ExpectationMaximisation

prod_parametrisation(::Marginalisation)         = ProdPreserveParametrisation()
prod_parametrisation(::ExpectationMaximisation) = ProdExpectationMaximisation()

## Common functions

function degree end

inbound_portal!(variable::AbstractVariable, portal) = error("Its not possible to change an inbound portal for $(variable)")

getmarginals(vector::AbstractVector{ <: AbstractVariable }) = collectLatest(map(getmarginal, vector))

getmarginal(variable::AbstractVariable) = getmarginal(variable, SkipInitial())

function getmarginal(variable::AbstractVariable, skip_strategy::MarginalSkipStrategy)
    vmarginal = _getmarginal(variable)
    if vmarginal === nothing
        vmarginal = as_marginal_observable(_makemarginal(variable))
        _setmarginal!(variable, vmarginal)
    end
    return as_marginal_observable(vmarginal, skip_strategy)
end

function setmarginal!(variable::AbstractVariable, marginal)
    setmarginal!(getmarginal(variable, IncludeAll()), marginal)
end

function setmarginals!(variables::AbstractVector{ <: AbstractVariable }, marginal)
    setmarginals!(variables, Iterators.repeated(marginal, length(variables)))
end

function setmarginals!(variables::AbstractVector{ <: AbstractVariable }, marginals::AbstractVector)
    foreach(zip(variables, marginals)) do (variable, marginal)
        setmarginal!(getmarginal(variable, IncludeAll()), marginal)
    end
end

function name(variable::AbstractVariable)
    return variable.name
end

## Helper functions

as_variable(x)                   = constvar(gensym(:as_var), x)
as_variable(v::AbstractVariable) = v
as_variable(t::Tuple)            = map(as_variable, t)


