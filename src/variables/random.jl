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

function randomvar(name::Symbol) 
    return RandomVariable(name, Vector{Union{Nothing, LazyObservable{Message}}}(), RandomVariableProps())
end

function randomvar(name::Symbol, dims::Tuple)
    return randomvar(name, dims...)
end

function randomvar(name::Symbol, dims::Vararg{Int})
    vars = Array{RandomVariable}(undef, dims)
    for index in CartesianIndices(axes(vars))
        @inbounds vars[index] = randomvar(Symbol(name, :_, Symbol(join(index.I, :_))))
    end
    return vars
end

degree(randomvar::RandomVariable) = length(randomvar.inputmsgs)

getlastindex(randomvar::RandomVariable) = length(randomvar.inputmsgs) + 1

messagein(randomvar::RandomVariable, index::Int)  = @inbounds randomvar.inputmsgs[index]
messageout(randomvar::RandomVariable, index::Int) = collectLatest(Message, Message, skipindex(randomvar.inputmsgs, index), __reduce_to_message)

_getmarginal(randomvar::RandomVariable)                                = randomvar.props.marginal
_setmarginal!(randomvar::RandomVariable, marginal::MarginalObservable) = randomvar.props.marginal = marginal
_makemarginal(randomvar::RandomVariable)                               = begin 
    # combineLatest(tuple(randomvar.inputmsgs...), PushEach()) |> discontinue() |> map(Marginal, __reduce_to_marginal)
    collectLatest(Message, Marginal, randomvar.inputmsgs, __reduce_to_marginal)
end

function setmessagein!(randomvar::RandomVariable, index::Int, messagein)
    if index === length(randomvar.inputmsgs) + 1
        push!(randomvar.inputmsgs, messagein)
    else
        throw("TODO error message: setmessagein! in ::RandomVariable")
    end
end