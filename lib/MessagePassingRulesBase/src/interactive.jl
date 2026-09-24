# Calling and querying rules by hand, at a REPL or in a notebook.

as_target(edge::Symbol) = Target(edge)
as_target((edge, k)::Tuple{Symbol, Integer}) = IndexedTarget(edge, k)
as_target(target::Union{Target, IndexedTarget}) = target
as_cluster(members::Tuple{Vararg{Symbol}}) = ClusterTarget(members)
as_cluster(target::ClusterTarget) = target

function interactive_args(m, q, clusters)
    isempty(clusters) && return RuleArgs(m = m, q = q)
    keys = Tuple(first.(clusters))
    values = Tuple(last.(clusters))
    return RuleArgs(as_messages(m), Marginals(q, Val(keys), values))
end

# Called with every rule an interactive call below selects, before it runs. Test tooling
# registers one to count a rule a test calls by hand as tested; an engine resolves rules
# itself and never reaches these.
const INTERACTIVE_SELECTION_OBSERVERS = Function[]

function selected_interactively(spec)
    foreach(observer -> observer(spec), INTERACTIVE_SELECTION_OBSERVERS)
    return spec
end

function call_resolved(spec, output, algorithm, ctx, args, ann, target)
    spec = selected_interactively(throw_if_not_found(spec))
    return execute_rule(spec, output, rule_algorithm(spec, algorithm), ctx, args, ann, target)
end

as_annotations(::Nothing) = NoAnnotations()
as_annotations(ann::RuleAnnotations) = ann
as_annotations(store) = RuleAnnotations(out = store)

"""
    call_message_update_rule(node, target; m = (;), q = (;), clusters = (), algorithm, ctx, ann)

Run the message rule of `node` towards `target` (`:out`, or `(:m, 2)` for a group member)
on the given inputs. `clusters` gives structural clusters as `(:y, :x) => value` pairs.
`algorithm` defaults to the node's; pass an `AnnotationStore` as `ann` to collect what the
rule annotates.
"""
function call_message_update_rule(node, target; m = NamedTuple(), q = NamedTuple(), clusters = (), algorithm = default_algorithm(node), ctx = RuleContext(), ann = nothing)
    resolved_target, args = as_target(target), interactive_args(m, q, clusters)
    return call_resolved(find_message_rule(node, resolved_target, algorithm, args), nothing, algorithm, ctx, args, as_annotations(ann), resolved_target)
end

"""
    call_marginal_update_rule(node, target; m, q, clusters, algorithm, ctx, ann)

As [`call_message_update_rule`](@ref), for the marginal of the cluster `target`, e.g. `(:out, :μ)`.
"""
function call_marginal_update_rule(node, target; m = NamedTuple(), q = NamedTuple(), clusters = (), algorithm = default_algorithm(node), ctx = RuleContext(), ann = nothing)
    cluster, args = as_cluster(target), interactive_args(m, q, clusters)
    return call_resolved(find_marginal_rule(node, cluster, algorithm, args), nothing, algorithm, ctx, args, as_annotations(ann), cluster)
end

"""
    call_average_energy(node; q, clusters, algorithm, ctx)

As [`call_message_update_rule`](@ref), for a node's average energy.
"""
function call_average_energy(node; m = NamedTuple(), q = NamedTuple(), clusters = (), algorithm = default_algorithm(node), ctx = RuleContext(), ann = nothing)
    args = interactive_args(m, q, clusters)
    return call_resolved(find_average_energy(node, algorithm, args), nothing, algorithm, ctx, args, as_annotations(ann), nothing)
end

"""
    which_message_update_rule(node, target; m, q, clusters, algorithm)

The [`RuleSpec`](@ref) that `call_message_update_rule` would run for these inputs; its display shows the
rule's source.
"""
function which_message_update_rule(node, target; m = NamedTuple(), q = NamedTuple(), clusters = (), algorithm = default_algorithm(node))
    resolved_target = as_target(target)
    return throw_if_not_found(find_message_rule(node, resolved_target, algorithm, interactive_args(m, q, clusters)))
end

"""
    which_marginal_update_rule(node, target; m, q, clusters, algorithm)

The [`RuleSpec`](@ref) that [`call_marginal_update_rule`](@ref) would run for these inputs.
"""
function which_marginal_update_rule(node, target; m = NamedTuple(), q = NamedTuple(), clusters = (), algorithm = default_algorithm(node))
    return throw_if_not_found(find_marginal_rule(node, as_cluster(target), algorithm, interactive_args(m, q, clusters)))
end

"""
    which_average_energy(node; m, q, clusters, algorithm)

The [`RuleSpec`](@ref) that [`call_average_energy`](@ref) would run for these inputs.
"""
function which_average_energy(node; m = NamedTuple(), q = NamedTuple(), clusters = (), algorithm = default_algorithm(node))
    return throw_if_not_found(find_average_energy(node, algorithm, interactive_args(m, q, clusters)))
end

function keyword_call(name, f, positional, args)
    keywords = Dict{Symbol, Any}()
    rest = Expr[]
    for arg in args
        (arg isa Expr && arg.head === :(=) && arg.args[1] isa Symbol) ||
            error("@$name takes keyword arguments only, like `node = ...`; got `$arg`")
        if arg.args[1] in positional
            keywords[arg.args[1]] = arg.args[2]
        else
            push!(rest, Expr(:kw, arg.args[1], arg.args[2]))
        end
    end
    for key in positional
        haskey(keywords, key) || error("@$name: `$key` is required")
    end
    return Expr(:call, f, Expr(:parameters, rest...), (keywords[key] for key in positional)...)
end

"""
    @call_message_update_rule(node = ..., target = ..., m = (...), q = (...), ...)

[`call_message_update_rule`](@ref), written with keywords.

```jldoctest
julia> struct Shift end

julia> @define_factor_node(node = Shift, type = Deterministic, interfaces = [:out, :in])

julia> @define_message_update_rule(
           node = Shift,
           target = :out,
           args = (m[:in]::Real,),
           body = (args) -> args.m[:in] + 1,
       )

julia> @call_message_update_rule(node = Shift, target = :out, m = (in = 1.0,))
2.0
```
"""
macro call_message_update_rule(args...)
    return esc(keyword_call("call_message_update_rule", call_message_update_rule, (:node, :target), args))
end

"""
    @call_marginal_update_rule(node = ..., target = (:y, :x), ...)

[`call_marginal_update_rule`](@ref), written with keywords.
"""
macro call_marginal_update_rule(args...)
    return esc(keyword_call("call_marginal_update_rule", call_marginal_update_rule, (:node, :target), args))
end

"""
    @call_average_energy(node = ..., q = (...), ...)

[`call_average_energy`](@ref), written with keywords.
"""
macro call_average_energy(args...)
    return esc(keyword_call("call_average_energy", call_average_energy, (:node,), args))
end

"""
    @which_message_update_rule(node = ..., target = ..., m = (...), ...)

[`which_message_update_rule`](@ref), written with keywords.
"""
macro which_message_update_rule(args...)
    return esc(keyword_call("which_message_update_rule", which_message_update_rule, (:node, :target), args))
end

target_edge_of(::Type{Target{E}}) where {E} = E
target_edge_of(::Type{IndexedTarget{E}}) where {E} = E
target_edge_of(::Type) = nothing

"""
    list_rules(node[, edge]; algorithm)

The rules defined for `node`: all of them, or the message rules towards `edge` (a group's
name for its members). With `algorithm`, only those it can select: its own, and for a
[`DefaultAlgorithmExtension`](@ref) the default's as well.
"""
function list_rules(node, edge::Union{Nothing, Symbol} = nothing; algorithm = nothing)
    return filter(registered_rules()) do spec
        node_matches(spec, node) &&
            (edge === nothing || (spec.kind === :message && target_edge_of(spec.target) === edge)) &&
            (algorithm === nothing || admits(algorithm, spec.algorithm))
    end
end

"""
    RuleCoverage

Which rules exist for a node: rows are message targets, marginal clusters and the average
energy; columns are algorithms; each cell counts rules. Made by [`rule_coverage`](@ref).
"""
struct RuleCoverage
    node::Any
    rows::Vector{String}
    algorithms::Vector{Type}
    counts::Dict{Tuple{String, Type}, Int}
end

target_label(::Type{Target{E}}) where {E} = ":$E"
target_label(::Type{IndexedTarget{E}}) where {E} = "(:$E, k)"
target_label(::Type{ClusterTarget{K}}) where {K} = repr(K)
target_label(::Type) = ""

function coverage_row(spec::RuleSpec)
    spec.kind === :average_energy && return "average energy"
    spec.kind === :marginal && return "q(" * join(spec.target.parameters[1], ", ") * ")"
    return spec.target <: IndexedTarget ? "→ ($(target_edge_of(spec.target)), k)" : "→ $(target_edge_of(spec.target))"
end

"""
    rule_coverage(node)

The [`RuleCoverage`](@ref) matrix of `node`: what can be computed, under which algorithm.
"""
function rule_coverage(node)
    specs = list_rules(node)
    rows = String[]
    algorithms = Type[]
    if applicable(nodespec, node)
        declared = nodespec(node)
        for interface in declared.interfaces
            push!(rows, interface.group ? "→ ($(interface.name), k)" : "→ $(interface.name)")
        end
        push!(algorithms, typeof(declared.algorithm))
    end
    counts = Dict{Tuple{String, Type}, Int}()
    for spec in specs
        row = coverage_row(spec)
        row in rows || push!(rows, row)
        spec.algorithm in algorithms || push!(algorithms, spec.algorithm)
        counts[(row, spec.algorithm)] = get(counts, (row, spec.algorithm), 0) + 1
    end
    return RuleCoverage(node, rows, algorithms, counts)
end

"""
    visualize_spec(spec)

Draw a [`NodeSpec`](@ref), [`RuleSpec`](@ref), [`DependenciesSpec`](@ref) or
[`RuleCoverage`](@ref). Implemented by visualisation backends, which are package
extensions; without one loaded this is a `MethodError` that says so.
"""
function visualize_spec end

"""
    @which_marginal_update_rule(node = ..., target = (:y, :x), ...)

[`which_marginal_update_rule`](@ref), written with keywords.
"""
macro which_marginal_update_rule(args...)
    return esc(keyword_call("which_marginal_update_rule", which_marginal_update_rule, (:node, :target), args))
end

"""
    @which_average_energy(node = ..., q = (...), ...)

[`which_average_energy`](@ref), written with keywords.
"""
macro which_average_energy(args...)
    return esc(keyword_call("which_average_energy", which_average_energy, (:node,), args))
end
