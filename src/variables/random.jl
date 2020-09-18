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