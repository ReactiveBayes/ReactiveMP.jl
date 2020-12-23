export randomvar, simplerandomvar

mutable struct RandomVariableProps
    marginal :: Union{Nothing, MarginalObservable}
    portal   :: AbstractPortal 

    RandomVariableProps() = new(nothing, EmptyPortal())
end

struct RandomVariable <: AbstractVariable
    name      :: Symbol
    inputmsgs :: Vector{LazyObservable{Message}}
    props     :: RandomVariableProps
end

function randomvar(name::Symbol) 
    return RandomVariable(name, Vector{LazyObservable{Message}}(), RandomVariableProps())
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

inbound_portal(randomvar::RandomVariable)          = randomvar.props.portal
inbound_portal!(randomvar::RandomVariable, portal) = randomvar.props.portal = portal

_getmarginal(randomvar::RandomVariable)                                = randomvar.props.marginal
_setmarginal!(randomvar::RandomVariable, marginal::MarginalObservable) = randomvar.props.marginal = marginal
_makemarginal(randomvar::RandomVariable)                               = collectLatest(Message, Marginal, randomvar.inputmsgs, __reduce_to_marginal)

function setmessagein!(randomvar::RandomVariable, index::Int, messagein)
    if index === length(randomvar.inputmsgs) + 1
        push!(randomvar.inputmsgs, messagein)
    else
        throw("TODO error message: setmessagein! in ::RandomVariable")
    end
end