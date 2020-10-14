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

degree(randomvar::RandomVariable) = length(randomvar.inputmsgs)

getlastindex(randomvar::RandomVariable) = length(randomvar.inputmsgs) + 1

messagein(randomvar::RandomVariable, index::Int)  = @inbounds randomvar.inputmsgs[index]
messageout(randomvar::RandomVariable, index::Int) = begin
    # TODO combineLatest is more efficient with small number of inputmsgs
    return collectLatest(Message, skipindex(randomvar.inputmsgs, index), Message, __reduce_to_message)
    # return combineLatest(skipindex(randomvar.inputmsgs, index)..., strategy = PushEach()) |> map(Message, __reduce_to_message)
end

_getmarginal(randomvar::RandomVariable)                                = randomvar.props.marginal
_setmarginal!(randomvar::RandomVariable, marginal::MarginalObservable) = randomvar.props.marginal = marginal
_makemarginal(randomvar::RandomVariable) = begin
    # TODO combineLatest is more efficient with small number of inputmsgs
    return collectLatest(Message, randomvar.inputmsgs, Marginal, __reduce_to_marginal)
    # return combineLatest(randomvar.inputmsgs..., strategy = PushEach()) |> map(Marginal, __reduce_to_marginal)
end

function setmessagein!(randomvar::RandomVariable, index::Int, messagein)
    if index === length(randomvar.inputmsgs) + 1
        push!(randomvar.inputmsgs, messagein)
    else
        throw("TODO error message: setmessagein! in ::RandomVariable")
    end
end