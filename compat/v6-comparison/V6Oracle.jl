# Calls a ReactiveMP v6 rule from the inputs a v7 rule takes, and returns its result and
# log scale. Only this environment has v6, so only this file names it.
module V6Oracle

using ReactiveMP

export v6_message_update, v6_logdensity, v6_interfaces

v6_target(towards::Symbol) = Val(towards)
v6_target((edge, k)::Tuple{Symbol, Integer}) = (Val(edge), k)

v6_name(key::Symbol) = key
v6_name(key::Tuple{Vararg{Symbol}}) = Symbol(join(key, "_"))

v6_wrap(wrap, value::Tuple) = ReactiveMP.ManyOf(map(v -> wrap(v, false, false), value))
v6_wrap(wrap, value) = wrap(value, false, false)

function v6_inputs(wrap, inputs::NamedTuple)
    isempty(inputs) && return nothing, nothing
    names = Val(map(v6_name, keys(inputs)))
    return names, map(v -> v6_wrap(wrap, v), Tuple(values(inputs)))
end

"""
    v6_message_update(fform, towards, m, q; meta = nothing)

Run v6's `rule` for `fform` towards `towards` on messages `m` and marginals `q`, given as
the named tuples a v7 rule takes. Returns `(result, logscale)`, `logscale` being `nothing`
when the rule annotates none.
"""
function v6_message_update(fform, towards, m::NamedTuple, q::NamedTuple; meta = nothing)
    mnames, messages = v6_inputs(ReactiveMP.Message, m)
    qnames, marginals = v6_inputs(ReactiveMP.Marginal, q)
    annotations = ReactiveMP.AnnotationDict()
    result = ReactiveMP.rule(
        fform, v6_target(towards), ReactiveMP.Marginalisation(), mnames, messages, qnames, marginals,
        meta, annotations, nothing,
    )
    result isa ReactiveMP.RuleMethodError && throw(result)
    logscale = ReactiveMP.has_annotation(annotations, :logscale) ? ReactiveMP.getlogscale(annotations) : nothing
    return result, logscale
end

"""The v6 node's log-density, as a keyword function of its interfaces."""
v6_logdensity(fform) = ReactiveMP.nodefunction(fform)

"""The v6 node's interface names."""
v6_interfaces(fform) = ReactiveMP.interfaces(fform) isa Val ? typeof(ReactiveMP.interfaces(fform)).parameters[1] : ReactiveMP.interfaces(fform)

end
