export randomvar, simplerandomvar

mutable struct RandomVariableProps
    marginal :: Union{Nothing, MarginalObservable}

    RandomVariableProps() = new(nothing)
end

struct RandomVariable{N} <: AbstractVariable
    name      :: Symbol
    inputmsgs :: Vector{Union{Nothing, LazyObservable{Message}}}
    props     :: RandomVariableProps
end

randomvar(name::Symbol, N::Int) = RandomVariable{N}(name, Vector{Union{Nothing, LazyObservable{Message}}}(undef, N), RandomVariableProps())

degree(::RandomVariable{N}) where N = N

messagein(randomvar::RandomVariable, index::Int)  = randomvar.inputmsgs[index]
messageout(randomvar::RandomVariable, index::Int) = begin
    return combineLatest(skipindex(randomvar.inputmsgs, index)..., strategy = PushNew()) |> reduce_to_message
end

_getmarginal(randomvar::RandomVariable)                                = randomvar.props.marginal
_setmarginal!(randomvar::RandomVariable, marginal::MarginalObservable) = randomvar.props.marginal = marginal
_makemarginal(randomvar::RandomVariable) = combineLatest(randomvar.inputmsgs..., strategy = PushNew()) |> reduce_to_marginal

function setmessagein!(randomvar::RandomVariable, index::Int, messagein)
    randomvar.inputmsgs[index] = messagein
    return nothing
end

##

mutable struct SimpleRandomVariableProps
    messagein1 :: Union{Nothing, LazyObservable{Message}}
    messagein2 :: Union{Nothing, LazyObservable{Message}}
    marginal   :: Union{Nothing, MarginalObservable}

    SimpleRandomVariableProps() = new(nothing, nothing, nothing)
end

struct SimpleRandomVariable <: AbstractVariable
    name  :: Symbol
    props :: SimpleRandomVariableProps
end

simplerandomvar(name::Symbol) = SimpleRandomVariable(name, SimpleRandomVariableProps())

degree(::SimpleRandomVariable) = 2

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

_getmarginal(srandomvar::SimpleRandomVariable)                                = srandomvar.props.marginal
_setmarginal!(srandomvar::SimpleRandomVariable, marginal::MarginalObservable) = srandomvar.props.marginal = marginal
_makemarginal(srandomvar::SimpleRandomVariable) = combineLatest(srandomvar.props.messagein1, srandomvar.props.messagein2, strategy = PushNew()) |> reduce_to_marginal

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
