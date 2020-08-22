export randomvar, simplerandomvar

mutable struct RandomVariableProps
    marginal :: Union{Nothing, MarginalObservable}

    RandomVariableProps() = new(nothing)
end

struct RandomVariable <: AbstractVariable
    name      :: Symbol
    inputmsgs :: Vector{Union{Nothing, LazyObservable{Message}}}
    props     :: RandomVariableProps
end

randomvar(name::Symbol) = RandomVariable(name, Vector{Union{Nothing, LazyObservable{Message}}}(), RandomVariableProps())

degree(::RandomVariable) = length(variable.inputmsgs)

getlastindex(randomvar::RandomVariable) = length(randomvar.inputmsgs) + 1

messagein(randomvar::RandomVariable, index::Int)  = @inbounds randomvar.inputmsgs[index]
messageout(randomvar::RandomVariable, index::Int) = begin
    return combineLatest(skipindex(randomvar.inputmsgs, index)..., strategy = PushNew()) |> reduce_to_message
end

_getmarginal(randomvar::RandomVariable)                                = randomvar.props.marginal
_setmarginal!(randomvar::RandomVariable, marginal::MarginalObservable) = randomvar.props.marginal = marginal
_makemarginal(randomvar::RandomVariable) = combineLatest(randomvar.inputmsgs..., strategy = PushNew()) |> reduce_to_marginal

function setmessagein!(randomvar::RandomVariable, index::Int, messagein)
    if index === length(randomvar.inputmsgs) + 1
        push!(randomvar.inputmsgs, messagein)
    else
        throw("TODO error message: setmessagein! in ::RandomVariable")
    end
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

function getlastindex(srandomvar::SimpleRandomVariable)
    if srandomvar.props.messagein1 === nothing
        return 1
    elseif srandomvar.props.messagein2 === nothing
        return 2
    else
        throw("TODO error message: getlastindex in ::SimpleRandomVariable")
    end
end

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
        @assert srandomvar.props.messagein1 === nothing
        srandomvar.props.messagein1 = messagein
    elseif index === 2
        @assert srandomvar.props.messagein2 === nothing
        srandomvar.props.messagein2 = messagein
    else
        error("Invalid `index`($(index)) in setmessagein! for SimpleRandomVariable object")
    end
    return nothing
end
