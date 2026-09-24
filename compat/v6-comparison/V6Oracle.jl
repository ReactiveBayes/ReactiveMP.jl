# Calls a ReactiveMP v6 rule from the inputs a v7 rule takes, and returns its result and
# log scale. Only this environment has v6, so only this file names it.
module V6Oracle

using ReactiveMP

export v6_message_update, v6_marginal_update, v6_average_energy, v6_logdensity, v6_interfaces, v6_cluster_blocks, v6_delta_node

v6_target(target::Symbol) = Val(target)
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
    v6_message_update(fform, target, m, q; meta = nothing, node = nothing)

Run v6's `rule` for `fform` target `target` on messages `m` and marginals `q`, given as
the named tuples a v7 rule takes. Returns `(result, logscale)`, `logscale` being `nothing`
when the rule annotates none. `node` is the v6 node a rule reads, such as a `DeltaFnNode`
holding the function ([`v6_delta_node`](@ref)).
"""
function v6_message_update(fform, target, m::NamedTuple, q::NamedTuple; meta = nothing, node = nothing)
    mnames, messages = v6_inputs(ReactiveMP.Message, m)
    qnames, marginals = v6_inputs(ReactiveMP.Marginal, q)
    annotations = ReactiveMP.AnnotationDict()
    result = ReactiveMP.rule(
        fform, v6_target(target), ReactiveMP.Marginalisation(), mnames, messages, qnames, marginals,
        meta, annotations, node,
    )
    result isa ReactiveMP.RuleMethodError && throw(result)
    logscale = ReactiveMP.has_annotation(annotations, :logscale) ? ReactiveMP.getlogscale(annotations) : nothing
    return result, logscale
end

"""
    v6_marginal_update(fform, members, m, q; meta = nothing, node = nothing)

Run v6's `marginalrule` for the cluster `members`, a tuple like `(:out, :μ)`, on messages `m`
and marginals `q`. v6 names the cluster by joining its members with `_`.
"""
function v6_marginal_update(fform, members::Tuple{Vararg{Symbol}}, m::NamedTuple, q::NamedTuple; meta = nothing, node = nothing)
    mnames, messages = v6_inputs(ReactiveMP.Message, m)
    qnames, marginals = v6_inputs(ReactiveMP.Marginal, q)
    return ReactiveMP.marginalrule(fform, Val(v6_name(members)), mnames, messages, qnames, marginals, meta, node)
end

"""
    v6_delta_node(f, meta)

The v6 `DeltaFnNode` for the function `f` that a Delta rule reads its function from, as v6's
`@call_rule` builds it, with the node type and meta the rule dispatches on.
"""
v6_delta_node(f, meta) = ReactiveMP.call_rule_make_node(ReactiveMP.DeltaFn{typeof(f)}, f, meta)

"""
    v6_average_energy(fform, q, clusters = (); meta = nothing)

Run v6's average energy for `fform` on marginals `q`, and on joint marginals given as
`members => value` pairs, in the order of the node's interfaces.
"""
function v6_average_energy(fform, q::NamedTuple, clusters = (); meta = nothing)
    inputs = merge(NamedTuple{map(v6_name, Tuple(first.(clusters)))}(Tuple(last.(clusters))), q)
    ordered = order_by_interfaces(fform, inputs)
    names, marginals = v6_inputs(ReactiveMP.Marginal, ordered)
    return ReactiveMP.score(ReactiveMP.AverageEnergy(), fform, names, marginals, meta)
end

# v6 dispatches on the names in the node's interface order, a joint `out_μ` by its first member.
function order_by_interfaces(fform, inputs::NamedTuple)
    names = collect(v6_interfaces(fform))
    position(key) = something(findfirst(==(Symbol(first(split(string(key), "_")))), names), length(names) + 1)
    sorted = sort(collect(keys(inputs)); by = position)
    return NamedTuple{Tuple(sorted)}(map(k -> inputs[k], Tuple(sorted)))
end

"""
    v6_cluster_blocks(fform, result::NamedTuple)

The blocks of a v6 marginal rule's NamedTuple result as `members => value` pairs, a key such
as `out_μ` split into `(:out, :μ)` by matching the node's own interface names, never by
guessing at `_`.
"""
function v6_cluster_blocks(fform, result::NamedTuple)
    names = map(string, collect(v6_interfaces(fform)))
    return [Tuple(split_into_interfaces(string(key), names)) => value for (key, value) in pairs(result)]
end

function split_into_interfaces(key::AbstractString, names)
    key in names && return [Symbol(key)]
    for name in names
        startswith(key, name * "_") || continue
        rest = split_into_interfaces(key[(ncodeunits(name) + 2):end], names)
        rest === nothing || return [Symbol(name); rest]
    end
    return nothing
end

"""The v6 node's log-density, as a keyword function of its interfaces."""
v6_logdensity(fform) = ReactiveMP.nodefunction(fform)

"""The v6 node's interface names."""
v6_interfaces(fform) = ReactiveMP.interfaces(fform) isa Val ? typeof(ReactiveMP.interfaces(fform)).parameters[1] : ReactiveMP.interfaces(fform)

end
