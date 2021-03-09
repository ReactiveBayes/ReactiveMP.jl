export AbstractVariable, degree
export RandomVariable, randomvar
export SimpleRandomVariable, simplerandomvar
export ConstVariable, constvar
export DataVariable, datavar, update!, finish!
export getmarginal, getmarginals, setmarginal!, activate!, name
export as_message, as_marginal
export as_variable

using Rocket

abstract type AbstractVariable end

function degree end

## Common functions

inbound_portal!(variable::AbstractVariable, portal) = error("Its not possible to change an inbound portal for $(variable)")

getmarginals(vector::AbstractVector{ <: AbstractVariable }) = collectLatest(map(getmarginal, vector))

getmarginal(variable::AbstractVariable) = getmarginal(SkipInitial(), variable)

function getmarginal(skip_strategy::Union{ SkipInitial, IncludeInitial }, variable::AbstractVariable)
    vmarginal = _getmarginal(variable)
    if vmarginal === nothing
        vmarginal = as_marginal_observable(IncludeInitial(), _makemarginal(variable))
        _setmarginal!(variable, vmarginal)
    end
    return as_marginal_observable(skip_strategy, vmarginal)
end

function setmarginal!(variable::AbstractVariable, marginal)
    setmarginal!(getmarginal(IncludeInitial(), variable), marginal)
end

function name(variable::AbstractVariable)
    return variable.name
end

## Helper functions

as_variable(x)                   = constvar(gensym(:as_var), x)
as_variable(v::AbstractVariable) = v
as_variable(t::Tuple)            = map(as_variable, t)


