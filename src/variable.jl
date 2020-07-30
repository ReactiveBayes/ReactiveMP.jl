export AbstractVariable, degree
export RandomVariable, randomvar
export SimpleRandomVariable, simplerandomvar
export ConstVariable, constvar
export DataVariable, datavar, update!, finish!
export getmarginal, setmarginal!, activate!, name
export as_message, as_marginal

using StaticArrays
using Rocket

abstract type AbstractVariable end

function degree end

## Common functions

function getmarginal(variable::AbstractVariable)
    vmarginal = _getmarginal(variable)
    if vmarginal === nothing
        nmarginal = MarginalObservable()
        connect!(nmarginal, _makemarginal(variable))
        _setmarginal!(variable, nmarginal)
        return nmarginal
    end
    return vmarginal
end

function setmarginal!(variable::AbstractVariable, marginal)
    setmarginal!(getmarginal(variable), marginal)
end

function name(variable::AbstractVariable)
    return variable.name
end

## RandomVariable
