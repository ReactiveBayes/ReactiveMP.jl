export randomvar, simplerandomvar

struct RandomVariable{N} <: AbstractVariable
    name      :: Symbol
    inputmsgs :: Vector{Union{Nothing, LazyObservable{AbstractMessage}}}
    belief    :: VariableBelief
end

# degree() # TODO

randomvar(name::Symbol, N::Int) = RandomVariable{N}(name, Vector{Union{Nothing, LazyObservable{AbstractMessage}}}(undef, N), VariableBelief())

messagein(randomvar::RandomVariable, index::Int)  = randomvar.inputmsgs[index]
messageout(randomvar::RandomVariable, index::Int) = begin
    return combineLatest(tuple(skipindex(randomvar.inputmsgs, index)...), true, (AbstractMessage, reduce_messages)) # TODO
end

makebelief(randomvar::RandomVariable)   = combineLatest(tuple(randomvar.inputmsgs...), true, (AbstractBelief, reduce_message_to_belief))

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
    name     :: Symbol
    props    :: SimpleRandomVariableProps
    belief   :: VariableBelief
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
