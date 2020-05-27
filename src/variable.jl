export AbstractVariable
export RandomVariable, randomvar
export SimpleRandomVariable, simplerandomvar
export ConstVariable, constvar
export DataVariable, datavar, update!, finish!
export getbelief, setbelief!, activate!, name

using StaticArrays
using Rocket

abstract type AbstractVariable end

## VariableBelief

struct VariableBelief{R}
    subject :: R
    stream  :: LazyObservable{AbstractBelief}
end

function VariableBelief()
    return VariableBelief(ReplaySubject(AbstractBelief, 1), lazy(AbstractBelief))
end

function connect!(belief::VariableBelief, source)
    set!(belief.stream, source |> multicast(belief.subject) |> ref_count())
    return nothing
end

function setbelief!(belief::VariableBelief, value::AbstractBelief)
    next!(belief.subject, value)
    return nothing
end

function getbelief(belief::VariableBelief)
    return belief.stream
end

## Common functions

function getbelief(variable::AbstractVariable)
    return getbelief(variable.belief)
end

function setbelief!(variable::AbstractVariable, value::AbstractBelief)
    setbelief!(variable.belief, value)
    return nothing
end

function activate!(variable::AbstractVariable)
    connect!(variable.belief, makebelief(variable))
    return nothing
end

function name(variable::AbstractVariable)
    return variable.name
end

## RandomVariable
