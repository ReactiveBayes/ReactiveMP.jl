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

## VariableMarginal

struct VariableMarginal{R}
    subject :: R
    stream  :: LazyObservable{AbstractMarginal}
end

function VariableMarginal()
    return VariableMarginal(ReplaySubject(AbstractMarginal, 1), lazy(AbstractMarginal))
end

function connect!(marginal::VariableMarginal, source)
    set!(marginal.stream, source |> multicast(marginal.subject) |> ref_count())
    return nothing
end

function setmarginal!(marginal::VariableMarginal, value)
    next!(marginal.subject, as_marginal(value))
    return nothing
end

function getmarginal(marginal::VariableMarginal)
    return marginal.stream
end

## Common functions

function getmarginal(variable::AbstractVariable)
    if !Rocket.isready(variable.marginal.stream)
        connect!(variable.marginal, makemarginal(variable))
    end
    return getmarginal(variable.marginal)
end

function setmarginal!(variable::AbstractVariable, value)
    setmarginal!(variable.marginal, value)
    return nothing
end

function name(variable::AbstractVariable)
    return variable.name
end

## RandomVariable
