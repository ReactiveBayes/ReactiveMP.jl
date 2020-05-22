export RandomVariable
export randomvar

using StaticArrays
using Rocket

struct RandomVariable{N}
    name      :: Symbol
    inputmsgs :: SVector{N, LazyObservable{Message}}
end

randomvar(name::Symbol, N::Int) = RandomVariable{N}(name, SVector{N}([ lazy(Message) for _ in 1:N ]))

messagein(randomvar::RandomVariable, index::Int)  = randomvar.inputmsgs[index]
messageout(randomvar::RandomVariable, index::Int) = begin
    return combineLatest(tuple(skipindex(randomvar.inputmsgs, index)...), true, (Message, reduce_messages)) # TODO
end

belief(randomvar::RandomVariable) = combineLatest(tuple(randomvar.inputmsgs...), true, (Belief, reduce_message_to_belief)) # TODO

##

struct ConstVariable{M}
    name       :: Symbol
    messageout :: M
end

constvar(name::Symbol, constval) = ConstVariable(name, of(constval))

messageout(constvar::ConstVariable, index::Int) = constvar.messageout
messagein(constvar::ConstVariable, index::Int)  = error("messagein is not defined for ConstVariable object")

belief(constvar::ConstVariable) = error("belief is not defined for ConstVariable object")

##

struct DataVariable{S}
    name      :: Symbol
    messagein :: S
end

function datavar(name::Symbol, ::Type{D}; subject = nothing) where D
    messagein = subject === nothing ? Subject(D) : subject
    return DataVariable(name, messagein)
end

messageout(datavar::DataVariable, index::Int) = error("messageout is not defined for DataVariable object")
messagein(datavar::DataVariable, index::Int)  = datavar.messagein

update!(datavar::DataVariable, data) = next!(messagein(datavar), data)

belief(datavar::DataVariable) = error("belief is not defined for DataVariable object")

##

struct PriorVariable{S}
    name       :: Symbol
    messageout :: S
end

function priorvar(name::Symbol, ::Type{D}; subject = nothing) where D
    messageout = subject === nothing ? Subject(D) : subject
    return PriorVariable(name, messageout)
end

messageout(priorvar::PriorVariable, index::Int) = priorvar.messageout
messagein(priorvar::PriorVariable, index::Int)  = error("messagein is not defined for PriorVariable object")

update!(priorvar::PriorVariable, prior) = next!(messageout(priorvar), prior)

belief(priorvar::PriorVariable) = error("belief is not defined for PriorVariable object")

##
