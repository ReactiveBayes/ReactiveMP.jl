export AbstractVariable
export RandomVariable, randomvar
export SimpleRandomVariable, simplerandomvar
export ConstVariable, constvar
export DataVariable, datavar, update!, finish!
export getbelief, setbelief!, activate!, name
export getmarginal

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

function setbelief!(belief::VariableBelief, value)
    next!(belief.subject, as_belief(value))
    return nothing
end

function getbelief(belief::VariableBelief)
    return belief.stream
end

## VariableMarginal

struct VariableMarginal{R}
    subject :: R
    stream  :: LazyObservable{AbstractBelief}
end

function VariableMarginal()
    return VariableMarginal(ReplaySubject(AbstractBelief, 1), lazy(AbstractBelief))
end

function connect!(marginal::VariableMarginal, source)
    set!(marginal.stream, source |> multicast(marginal.subject) |> ref_count())
    return nothing
end

function getmarginal(marginal::VariableMarginal)
    return marginal.stream
end

## Common functions

function getbelief(variable::AbstractVariable)
    if !Rocket.isready(variable.belief.stream)
        connect!(variable.belief, makebelief(variable))
    end
    return getbelief(variable.belief)
end

function setbelief!(variable::AbstractVariable, value)
    setbelief!(variable.belief, value)
    return nothing
end

function getmarginal(variable::AbstractVariable)
    if !Rocket.isready(variable.marginal.stream)
        connect!(variable.marginal, makemarginal(variable))
    end
    return getmarginal(variable.marginal)
end

function name(variable::AbstractVariable)
    return variable.name
end

## RandomVariable
