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

as_annotations(::Nothing) = NoAnnotations()
as_annotations(ann::RuleAnnotations) = ann
as_annotations(store) = RuleAnnotations(out = store)

"""
    call_rule(node, towards; m = (;), q = (;), clusters = (), algorithm, ctx, ann)

Run the message rule of `node` towards `towards` (`:out`, or `(:m, 2)` for a group member)
on the given inputs. `clusters` gives structural clusters as `(:y, :x) => value` pairs.
`algorithm` defaults to the node's; pass an `AnnotationStore` as `ann` to collect what the
rule annotates.
"""
call_rule(node, towards; m = NamedTuple(), q = NamedTuple(), clusters = (), algorithm = default_algorithm(node), ctx = RuleContext(), ann = nothing) =
    message_passing_rule(node, as_target(towards), algorithm, interactive_args(m, q, clusters), ctx, as_annotations(ann))

"""
    call_marginalrule(node, towards; m, q, clusters, algorithm, ctx, ann)

As [`call_rule`](@ref), for the marginal of the cluster `towards`, e.g. `(:out, :μ)`.
"""
call_marginalrule(node, towards; m = NamedTuple(), q = NamedTuple(), clusters = (), algorithm = default_algorithm(node), ctx = RuleContext(), ann = nothing) =
    message_passing_marginalrule(node, as_cluster(towards), algorithm, interactive_args(m, q, clusters), ctx, as_annotations(ann))

"""
    call_average_energy(node; q, clusters, algorithm, ctx)

As [`call_rule`](@ref), for a node's average energy.
"""
call_average_energy(node; m = NamedTuple(), q = NamedTuple(), clusters = (), algorithm = default_algorithm(node), ctx = RuleContext(), ann = nothing) =
    message_passing_average_energy(node, algorithm, interactive_args(m, q, clusters), ctx, as_annotations(ann))

"""
    which_rule(node, towards; m, q, clusters, algorithm)

The [`RuleSpec`](@ref) that `call_rule` would run for these inputs; its display shows the
rule's source.
"""
function which_rule(node, towards; m = NamedTuple(), q = NamedTuple(), clusters = (), algorithm = default_algorithm(node))
    target = as_target(towards)
    return throw_if_not_found(find_message_rule(node, target, algorithm, interactive_args(m, q, clusters)))
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
    @call_rule(node = ..., towards = ..., m = (...), q = (...), ...)

[`call_rule`](@ref), written with keywords.

```julia
@call_rule(node = NormalMeanVariance, towards = :out, m = (μ = PointMass(0.0), v = PointMass(1.0)))
```
"""
macro call_rule(args...)
    return esc(keyword_call("call_rule", call_rule, (:node, :towards), args))
end

"""
    @call_marginalrule(node = ..., towards = (:y, :x), ...)

[`call_marginalrule`](@ref), written with keywords.
"""
macro call_marginalrule(args...)
    return esc(keyword_call("call_marginalrule", call_marginalrule, (:node, :towards), args))
end

"""
    @call_average_energy(node = ..., q = (...), ...)

[`call_average_energy`](@ref), written with keywords.
"""
macro call_average_energy(args...)
    return esc(keyword_call("call_average_energy", call_average_energy, (:node,), args))
end

"""
    @which_rule(node = ..., towards = ..., m = (...), ...)

[`which_rule`](@ref), written with keywords.
"""
macro which_rule(args...)
    return esc(keyword_call("which_rule", which_rule, (:node, :towards), args))
end

target_edge_of(::Type{Target{E}}) where {E} = E
target_edge_of(::Type{IndexedTarget{E}}) where {E} = E
target_edge_of(::Type) = nothing

"""
    list_rules(node[, edge]; algorithm)

The rules defined for `node`: all of them, or the message rules towards `edge` (a group's
name for its members). With `algorithm`, only those it selects.
"""
function list_rules(node, edge::Union{Nothing, Symbol} = nothing; algorithm = nothing)
    return filter(registered_rules()) do spec
        node_matches(spec, node) &&
            (edge === nothing || (spec.kind === :message && target_edge_of(spec.target) === edge)) &&
            (algorithm === nothing || algorithm isa spec.algorithm)
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
