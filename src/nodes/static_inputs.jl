# Static inputs: the members of a node's group connected to a constant or to data, folded into
# its function under `static_inputs = :fold`.

"""
    ReactiveMP.StaticFold(f, statics::Tuple)

The function of a node that folds its static inputs (`static_inputs = :fold`), which
[`factornode`](@ref) builds from its `nodefn`. Called with the free inputs, it calls `f` with every
input in its place, each static one at its position `k` taking the latest value of its variable:
the constant, or the last observation. `statics` holds `(k, variable)` pairs.
"""
struct StaticFold{F, P, V, N}
    f::F
    variables::V
end

function StaticFold(f::F, statics::Tuple) where {F}
    positions = map(first, statics)
    variables = map(last, statics)
    return StaticFold{F, positions, typeof(variables), length(statics)}(f, variables)
end

static_variables(fold::StaticFold) = fold.variables
static_variables(::Any) = ()

static_value(variable::ConstVariable) = getconst(variable)
static_value(variable::DataVariable) = BayesBase.getpointmass(getdata(Rocket.getrecent(get_stream_of_outbound_messages(variable, 1))))

@generated function (fold::StaticFold{F, P, V, N})(free...) where {F, P, V, N}
    total = length(free) + N
    next = 0
    arguments = map(1:total) do k
        position = findfirst(==(k), P)
        position === nothing ? :(free[$(next += 1)]) : :(static_value(fold.variables[$position]))
    end
    return :(fold.f($(arguments...)))
end

function MessagePassingRulesBase.getnodefn(factornode::FactorNode, ::MessagePassingRulesBase.Target{:out})
    factornode.nodefn === nothing && throw(ArgumentError("`$(functionalform(factornode))` was created without `nodefn`, so it has no function to give a rule"))
    return factornode.nodefn
end

"""
    ReactiveMP.with_statics(factornode, stream)

`stream`, gated on the node's static inputs: it emits only once each of them has a value, and
again when one of them updates. Without static inputs it is `stream` itself.
"""
with_statics(factornode, stream) = with_statics(static_variables(factornode.nodefn), stream)
with_statics(::Tuple{}, stream) = stream

function with_statics(variables::Tuple, stream::T) where {T}
    statics = map(variable -> get_stream_of_outbound_messages(variable, 1), variables)
    return combineLatest((stream, combineLatest(statics, PushNew()))) |> map(eltype(T), first)
end
