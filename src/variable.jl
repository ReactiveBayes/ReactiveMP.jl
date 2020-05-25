export AbstractVariable
export RandomVariable, randomvar
export ConstVariable, constvar
export DataVariable, datavar
export belief

# TODO typed fauled?

using StaticArrays
using Rocket

abstract type AbstractVariable end

struct RandomVariable{N} <: AbstractVariable
    name      :: Symbol
    inputmsgs :: SVector{N, LazyObservable{AbstractMessage}}
end

randomvar(name::Symbol, N::Int) = RandomVariable{N}(name, SVector{N}([ lazy(AbstractMessage) for _ in 1:N ]))

messagein(randomvar::RandomVariable, index::Int)  = randomvar.inputmsgs[index]
messageout(randomvar::RandomVariable, index::Int) = begin
    return combineLatest(tuple(skipindex(randomvar.inputmsgs, index)...), true, (AbstractMessage, reduce_messages)) # TODO
end

belief(randomvar::RandomVariable) = combineLatest(tuple(randomvar.inputmsgs...), true, (Belief, reduce_message_to_belief)) # TODO

##

struct ConstVariable{M} <: AbstractVariable
    name       :: Symbol
    messageout :: M
    messagein  :: LazyObservable{AbstractMessage}
end

constvar(name::Symbol, constval) = ConstVariable(name, of(Message(constval)), lazy(AbstractMessage))

function messageout(constvar::ConstVariable, index::Int)
    @assert index === 1
    return constvar.messageout
end

function messagein(constvar::ConstVariable, index::Int)
    @assert index === 1
    return constvar.messagein
end

belief(constvar::ConstVariable) = combineLatest((messageout(constvar, 1), messagein(constvar, 1)), true, (AbstractBelief, reduce_message_to_belief))

##

struct DataVariable{S, D} <: AbstractVariable
    name       :: Symbol
    messageout :: S
    messagein  :: LazyObservable
end

function datavar(name::Symbol, ::Type{D}; subject::S = Subject(Message{D})) where { S, D }
    return DataVariable{S, D}(name, subject, lazy(AbstractMessage))
end

function messageout(datavar::DataVariable, index::Int)
    @assert index === 1
    return datavar.messageout
end

function messagein(datavar::DataVariable, index::Int)
    @assert index === 1
    return datavar.messagein
end

update!(datavar::DataVariable{S, D}, data::D) where { S, D } = next!(messageout(datavar), Message(data))

belief(datavar::DataVariable) = combineLatest((messageout(constvar, 1), messagein(constvar, 1)), true, (AbstractBelief, reduce_message_to_belief))
