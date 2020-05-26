export AbstractVariable
export RandomVariable, randomvar
export SimpleRandomVariable, simplerandomvar
export ConstVariable, constvar
export DataVariable, datavar, update!, finish!
export getbelief

using StaticArrays
using Rocket

abstract type AbstractVariable end

## VariableBelief

struct VariableBelief{R}
    subject :: R
    belief  :: LazyObservable{AbstractBelief}
end

function VariableBelief()
    return VariableBelief(ReplaySubject(AbstractBelief, 1), lazy(AbstractBelief))
end

function connect!(belief::VariableBelief, source)
    set!(belief.belief, source |> multicast(belief.subject) |> ref_count())
end

function setbelief!(belief::VariableBelief, value::AbstractBelief)
    next!(belief.subject, value)
end

function getbelief(belief::VariableBelief)
    return belief.belief
end

## Common functions

function getbelief(variable::AbstractVariable)
    return getbelief(variable.belief)
end

function activate!(variable::AbstractVariable)
    connect!(variable.belief, makebelief(variable))
end

## RandomVariable

struct RandomVariable{N} <: AbstractVariable
    name      :: Symbol
    inputmsgs :: Vector{Union{Nothing, LazyObservable{AbstractMessage}}}
    belief    :: VariableBelief
end

randomvar(name::Symbol, N::Int) = RandomVariable{N}(name, Vector{Union{Nothing, LazyObservable{AbstractMessage}}}(undef, N), VariableBelief())

messagein(randomvar::RandomVariable, index::Int)  = randomvar.inputmsgs[index]
messageout(randomvar::RandomVariable, index::Int) = begin
    return combineLatest(tuple(skipindex(randomvar.inputmsgs, index)...), true, (AbstractMessage, reduce_messages)) # TODO
end

makebelief(randomvar::RandomVariable) = combineLatest(tuple(randomvar.inputmsgs...), true, (AbstractBelief, reduce_message_to_belief)) # TODO

function setmessagein!(randomvar::RandomVariable, index::Int, messagein)
    randomvar.inputmsgs[index] = messagein
    return nothing
end

##

mutable struct SimpleRandomVariableProps
    messagein1 :: Union{Nothing, LazyObservable{AbstractMessage}}
    messagein2 :: Union{Nothing, LazyObservable{AbstractMessage}}

    SimpleRandomVariableProps() = new(nothing, nothing)
end

struct SimpleRandomVariable <: AbstractVariable
    name   :: Symbol
    props  :: SimpleRandomVariableProps
    belief :: VariableBelief
end

simplerandomvar(name::Symbol) = SimpleRandomVariable(name, SimpleRandomVariableProps(), VariableBelief())

function messagein(srandomvar::SimpleRandomVariable, index::Int)
    if index === 1
        return srandomvar.props.messagein1
    elseif index === 2
        return srandomvar.props.messagein2
    else
        error("Invalid `index`($(index)) in messagein for SimpleRandomVariable object")
    end
end

function messageout(srandomvar::SimpleRandomVariable, index::Int)
    if index === 2
        return srandomvar.props.messagein1
    elseif index === 1
        return srandomvar.props.messagein2
    else
        error("Invalid `index`($(index)) in messageout for SimpleRandomVariable object")
    end
end

makebelief(srandomvar::SimpleRandomVariable) = combineLatest((srandomvar.props.messagein1, srandomvar.props.messagein2), true, (AbstractBelief, reduce_message_to_belief))

function setmessagein!(srandomvar::SimpleRandomVariable, index::Int, messagein)
    if index === 1
        srandomvar.props.messagein1 = messagein
    elseif index === 2
        srandomvar.props.messagein2 = messagein
    else
        error("Invalid `index`($(index)) in setmessagein! for SimpleRandomVariable object")
    end
    return nothing
end

##

mutable struct ConstVariableProps
    messagein :: Union{Nothing, LazyObservable{AbstractMessage}}

    ConstVariableProps() = new(nothing)
end

struct ConstVariable{M} <: AbstractVariable
    name       :: Symbol
    messageout :: M
    props      :: ConstVariableProps
    belief     :: VariableBelief
end

constvar(name::Symbol, constval) = ConstVariable(name, of(Message(constval)), ConstVariableProps(), VariableBelief())

function messageout(constvar::ConstVariable, index::Int)
    @assert index === 1
    return constvar.messageout
end

function messagein(constvar::ConstVariable, index::Int)
    @assert index === 1
    return constvar.props.messagein
end

makebelief(constvar::ConstVariable) = combineLatest((constvar.messageout, constvar.props.messagein), true, (AbstractBelief, reduce_message_to_belief))

function setmessagein!(constvar::ConstVariable, index::Int, messagein)
    @assert index === 1
    constvar.props.messagein = messagein
    return nothing
end

##

mutable struct DataVariableProps
    messagein :: Union{Nothing, LazyObservable{AbstractMessage}}

    DataVariableProps() = new(nothing)
end

struct DataVariable{S, D} <: AbstractVariable
    name       :: Symbol
    messageout :: S
    props      :: DataVariableProps
    belief     :: VariableBelief
end

function datavar(name::Symbol, ::Type{D}; subject::S = Subject(Message{D})) where { S, D }
    return DataVariable{S, D}(name, subject, DataVariableProps(), VariableBelief())
end

function messageout(datavar::DataVariable, index::Int)
    @assert index === 1
    return datavar.messageout
end

function messagein(datavar::DataVariable, index::Int)
    @assert index === 1
    return datavar.props.messagein
end

update!(datavar::DataVariable{S, D}, data::D) where { S, D } = next!(messageout(datavar, 1), Message(data))
finish!(datavar::DataVariable) = complete!(messageout(datavar, 1))

makebelief(datavar::DataVariable) = combineLatest((datavar.messageout, datavar.props.messagein), true, (AbstractBelief, reduce_message_to_belief))

function setmessagein!(datavar::DataVariable, index::Int, messagein)
    @assert index === 1
    datavar.props.messagein = messagein
    return nothing
end
