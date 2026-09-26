# Calling and querying rules by hand, at a REPL or in a notebook.

as_target(edge::Symbol) = Target(edge)
as_target((edge, k)::Tuple{Symbol, Integer}) = IndexedTarget(edge, k)
as_target(target::Union{Target, IndexedTarget}) = target
as_cluster(members::Tuple{Vararg{ClusterMember}}) = ClusterTarget(members)
as_cluster(target::ClusterTarget) = target

function interactive_args(m, q, clusters, logscale = nothing)
    isempty(clusters) && return RuleArgs(m = m, q = q, logscale = logscale)
    keys = Tuple(first.(clusters))
    values = Tuple(last.(clusters))
    return RuleArgs(as_messages(m), Marginals(q, Val(keys), values), as_logscales(logscale))
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
    return run_rule(spec, output, rule_algorithm(spec, algorithm), ctx, args, ann, target)
end

as_annotations(::Nothing) = NoAnnotations()
as_annotations(ann::RuleAnnotations) = ann
as_annotations(store) = RuleAnnotations(out = store)

# Parts of the call and query docstrings that several share, written once and interpolated.

const DOC_CALL_NODE = rstrip(
    """
    - `node`: the node, as declared with [`@define_factor_node`](@ref): a type,
      `NormalMeanVariance`, or a function, `+`.
    """
)

const DOC_CALL_MESSAGE_TARGET = rstrip(
    """
    - `target`: the interface the message goes to: `:out`, or `(:m, k)` for member `k` of the group
      `m`, as `(:in, 2)`.
    """
)

const DOC_CALL_MARGINAL_TARGET = rstrip(
    """
    - `target`: the cluster whose joint marginal to compute, its members in interface order:
      `(:out, :μ)`, or `(:out, (:T, 1))` with a group member.
    """
)

const DOC_CALL_INPUTS = rstrip(
    """
    - `m`: the inbound messages, a `NamedTuple` keyed by the interfaces' declared names:
      `m = (μ = NormalMeanVariance(0.0, 1.0), v = PointMass(1.0))`. A group is a tuple of its members
      in order, `m = (in = (m₁, m₂),)`; for a rule that selects some members, `nothing` may stand
      for the others, as an engine gives them. Default: none.

    - `q`: the marginals of single interfaces, keyed the same way: `q = (v = Gamma(1.0, 1.0),)`.
      Default: none.

    - `clusters`: the joint marginals of structural clusters, a tuple of pairs from the cluster's
      members, in interface order, to its joint: `clusters = ((:y, :x) => q_yx,)`. `(:in,) => q` is
      the joint over the group `in`. Default: none.
    """
)

const DOC_CALL_ALGORITHM = rstrip(
    """
    - `algorithm`: the algorithm value to run under, with its parameters:
      `algorithm = ARVMP(Multivariate, 2, ARsafe())`. Rule lookup selects the rules of its type, and
      for a [`DefaultAlgorithmExtension`](@ref) the default's as well; an inherited rule runs with
      the value it was written for ([`rule_algorithm`](@ref)). Default: the node's,
      [`default_algorithm`](@ref)`(node)`.
    """
)

const DOC_CALL_CTX = rstrip(
    """
    - `ctx`: the [`RuleContext`](@ref) of services the rule reads as `ctx.name`:
      `ctx = MessagePassingRulesBase.RuleContext(rng = Xoshiro(1))`. Default: an empty context. **Its services are
      not checked**: one the rule declares and `ctx` does not supply reads as `nothing` inside the
      rule, so an unset `matrix_correction` gives the rule's own default. An engine checks them when
      it resolves a rule; call [`MessagePassingRulesBase.check_services`](@ref check_services)`(getrule(result), ctx)`
      for the same guarantee.
    """
)

const DOC_CALL_ANN = rstrip(
    """
    - `ann`: where the annotations go. Default: `nothing`, and what the rule annotates is dropped.
      A [`MessagePassingRulesBase.AnnotationStore`](@ref AnnotationStore)`()` collects what the rule
      writes with [`annotate!`](@ref), returned by [`getannotations`](@ref)`(result)`. A
      [`MessagePassingRulesBase.RuleAnnotations`](@ref RuleAnnotations)`(m = …, q = …, out = AnnotationStore())`
      also gives the annotations that arrived with the inputs, which the rule reads as `ann.m[:μ]`.
    """
)

const DOC_CALL_ERRORS = rstrip(
    """
    When no rule fits, the call throws a [`RuleNotFoundError`](@ref), which lists the closest rules
    and, input by input, why each does not fit. An unknown keyword is an error. Test tooling counts
    a rule called this way as tested, for its rule-coverage gate.
    """
)

"""
    call_message_update_rule(node, target; m, q, clusters, logscale, algorithm, ctx, ann)

Run the message rule of `node` towards `target` on the inputs given, as an engine would, and
return a [`RuleResult`](@ref): [`getresult`](@ref) is the message, [`getlogscale`](@ref) its log
scale. For exploring a rule at the REPL, in a notebook or in a test;
[`@call_message_update_rule`](@ref) is the same call written with keywords only.

# Arguments

$(DOC_CALL_NODE)

$(DOC_CALL_MESSAGE_TARGET)

# Keywords

All optional.

$(DOC_CALL_INPUTS)

- `logscale`: the log scales that arrived with the messages, keyed like `m`: `logscale = (μ =
  0.0,)`. Needed only by a rule declared with `reads_logscale = true`, for which the call is an
  error without them. Default: `nothing`.

$(DOC_CALL_ALGORITHM)

$(DOC_CALL_CTX)

$(DOC_CALL_ANN)

$(DOC_CALL_ERRORS)
"""
function call_message_update_rule(node, target; m = NamedTuple(), q = NamedTuple(), clusters = (), logscale = nothing, algorithm = default_algorithm(node), ctx = RuleContext(), ann = nothing)
    resolved_target, args = as_target(target), interactive_args(m, q, clusters, logscale)
    return call_resolved(find_message_rule(node, resolved_target, algorithm, args), nothing, algorithm, ctx, args, as_annotations(ann), resolved_target)
end

"""
    call_marginal_update_rule(node, target; m, q, clusters, algorithm, ctx, ann)

Run the marginal rule of `node` for the cluster `target` on the inputs given, as an engine
would, and return a [`RuleResult`](@ref): [`getresult`](@ref) is the joint marginal, and its log
scale is `nothing`. [`@call_marginal_update_rule`](@ref) is the same call written with keywords
only.

# Arguments

$(DOC_CALL_NODE)

$(DOC_CALL_MARGINAL_TARGET)

# Keywords

All optional. A marginal rule typically reads the messages on the cluster's members and the
marginals of the node's other interfaces.

$(DOC_CALL_INPUTS)

$(DOC_CALL_ALGORITHM)

$(DOC_CALL_CTX)

$(DOC_CALL_ANN)

$(DOC_CALL_ERRORS)
"""
function call_marginal_update_rule(node, target; m = NamedTuple(), q = NamedTuple(), clusters = (), algorithm = default_algorithm(node), ctx = RuleContext(), ann = nothing)
    cluster, args = as_cluster(target), interactive_args(m, q, clusters)
    return call_resolved(find_marginal_rule(node, cluster, algorithm, args), nothing, algorithm, ctx, args, as_annotations(ann), cluster)
end

"""
    call_average_energy(node; q, clusters, m, algorithm, ctx, ann)

Compute the average energy of `node` under the marginals given, as an engine's free energy
would, and return a [`RuleResult`](@ref): [`getresult`](@ref) is the energy, a number, and its
log scale is `nothing`. [`@call_average_energy`](@ref) is the same call written with keywords
only.

# Arguments

$(DOC_CALL_NODE)

# Keywords

All optional. An average energy reads marginals, `q` and `clusters`, one per cluster of the
factorisation.

$(DOC_CALL_INPUTS)

$(DOC_CALL_ALGORITHM)

$(DOC_CALL_CTX)

$(DOC_CALL_ANN)

$(DOC_CALL_ERRORS)
"""
function call_average_energy(node; m = NamedTuple(), q = NamedTuple(), clusters = (), algorithm = default_algorithm(node), ctx = RuleContext(), ann = nothing)
    args = interactive_args(m, q, clusters)
    return call_resolved(find_average_energy(node, algorithm, args), nothing, algorithm, ctx, args, as_annotations(ann), nothing)
end

"""
    which_message_update_rule(node, target; m, q, clusters, algorithm)

The [`RuleSpec`](@ref) that [`call_message_update_rule`](@ref) would run for these inputs,
without running it. It displays the rule with its source, file and line.
[`@which_message_update_rule`](@ref) is the same query written with keywords only.

# Arguments

$(DOC_CALL_NODE)

$(DOC_CALL_MESSAGE_TARGET)

# Keywords

All optional; only the inputs' types matter, as they are what selects a rule.

$(DOC_CALL_INPUTS)

$(DOC_CALL_ALGORITHM)

When no rule fits, the query throws a [`RuleNotFoundError`](@ref), which lists the closest rules
and why each does not fit.
"""
function which_message_update_rule(node, target; m = NamedTuple(), q = NamedTuple(), clusters = (), algorithm = default_algorithm(node))
    resolved_target = as_target(target)
    return throw_if_not_found(find_message_rule(node, resolved_target, algorithm, interactive_args(m, q, clusters)))
end

"""
    which_marginal_update_rule(node, target; m, q, clusters, algorithm)

The [`RuleSpec`](@ref) that [`call_marginal_update_rule`](@ref) would run for these inputs,
without running it. It displays the rule with its source, file and line.
[`@which_marginal_update_rule`](@ref) is the same query written with keywords only.

# Arguments

$(DOC_CALL_NODE)

$(DOC_CALL_MARGINAL_TARGET)

# Keywords

All optional; only the inputs' types matter, as they are what selects a rule.

$(DOC_CALL_INPUTS)

$(DOC_CALL_ALGORITHM)

When no rule fits, the query throws a [`RuleNotFoundError`](@ref), which lists the closest rules
and why each does not fit.
"""
function which_marginal_update_rule(node, target; m = NamedTuple(), q = NamedTuple(), clusters = (), algorithm = default_algorithm(node))
    return throw_if_not_found(find_marginal_rule(node, as_cluster(target), algorithm, interactive_args(m, q, clusters)))
end

"""
    which_average_energy(node; q, clusters, m, algorithm)

The [`RuleSpec`](@ref) that [`call_average_energy`](@ref) would run for these marginals, without
running it. It displays the energy with its source, file and line.
[`@which_average_energy`](@ref) is the same query written with keywords only.

# Arguments

$(DOC_CALL_NODE)

# Keywords

All optional; only the inputs' types matter, as they are what selects the energy.

$(DOC_CALL_INPUTS)

$(DOC_CALL_ALGORITHM)

When no energy fits, the query throws a [`RuleNotFoundError`](@ref), which lists the closest
ones and why each does not fit.
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
    @call_message_update_rule(
        node = ..., target = ..., m = (...), q = (...), clusters = (...),
        logscale = (...), algorithm = ..., ctx = ..., ann = ...,
    )

Run the message rule of a node towards one of its interfaces on the inputs given, as an engine
would, and return a [`RuleResult`](@ref): [`getresult`](@ref) is the message,
[`getlogscale`](@ref) its log scale. For exploring a rule at the REPL, in a notebook or in a
test. The macro takes keyword arguments only; `node` and `target` are required, the rest
optional. It is [`call_message_update_rule`](@ref) with `node` and `target` given by name.

# Required keywords

$(DOC_CALL_NODE)

$(DOC_CALL_MESSAGE_TARGET)

# Optional keywords

$(DOC_CALL_INPUTS)

- `logscale`: the log scales that arrived with the messages, keyed like `m`: `logscale = (μ =
  0.0,)`. Needed only by a rule declared with `reads_logscale = true`, for which the call is an
  error without them. Default: `nothing`.

$(DOC_CALL_ALGORITHM)

$(DOC_CALL_CTX)

$(DOC_CALL_ANN)

# The result

A [`RuleResult`](@ref): [`getresult`](@ref) the message; [`getlogscale`](@ref) its log scale, a
number, or an [`UndefinedLogScale`](@ref) when the rule declares none; [`getrule`](@ref) the
[`RuleSpec`](@ref) that ran; [`getannotations`](@ref) what it annotated. In the terminal it shows
a report of the call, one line per edge of the node, and in a notebook a card with the node
drawn.

$(DOC_CALL_ERRORS)

# Example

```jldoctest
julia> struct Shift end

julia> @define_factor_node(node = Shift, type = Deterministic, interfaces = [:out, :in])

julia> @define_message_update_rule(
           node = Shift,
           target = :out,
           args = (m[:in]::Real,),
           logscale = 0,
           body = (args) -> args.m[:in] + 1,
       )

julia> result = @call_message_update_rule(node = Shift, target = :out, m = (in = 1.0,));

julia> getresult(result), getlogscale(result)
(2.0, 0)
```
"""
macro call_message_update_rule(args...)
    return esc(keyword_call("call_message_update_rule", call_message_update_rule, (:node, :target), args))
end

"""
    @call_marginal_update_rule(
        node = ..., target = (:y, :x), m = (...), q = (...), clusters = (...),
        algorithm = ..., ctx = ..., ann = ...,
    )

Run the marginal rule of a node for one of its clusters on the inputs given, as an engine would,
and return a [`RuleResult`](@ref): [`getresult`](@ref) is the joint marginal. The macro takes
keyword arguments only; `node` and `target` are required, the rest optional. It is
[`call_marginal_update_rule`](@ref) with `node` and `target` given by name. A marginal has no
log scale, so `logscale` is not accepted.

# Required keywords

$(DOC_CALL_NODE)

$(DOC_CALL_MARGINAL_TARGET)

# Optional keywords

A marginal rule typically reads the messages on the cluster's members and the marginals of the
node's other interfaces.

$(DOC_CALL_INPUTS)

$(DOC_CALL_ALGORITHM)

$(DOC_CALL_CTX)

$(DOC_CALL_ANN)

# The result

A [`RuleResult`](@ref): [`getresult`](@ref) the joint marginal; [`getlogscale`](@ref) is
`nothing`; [`getrule`](@ref) the [`RuleSpec`](@ref) that ran; [`getannotations`](@ref) what it
annotated.

$(DOC_CALL_ERRORS)

# Example

```julia
result = @call_marginal_update_rule(
    node   = NormalMeanVariance,
    target = (:out, :μ),
    m      = (out = NormalMeanVariance(1.0, 1.0), μ = NormalMeanVariance(0.0, 2.0)),
    q      = (v = PointMass(1.0),),
)
getresult(result)
```
"""
macro call_marginal_update_rule(args...)
    return esc(keyword_call("call_marginal_update_rule", call_marginal_update_rule, (:node, :target), args))
end

"""
    @call_average_energy(node = ..., q = (...), clusters = (...), m = (...), algorithm = ..., ctx = ..., ann = ...)

Compute the average energy of a node under the marginals given, as an engine's free energy
would, and return a [`RuleResult`](@ref): [`getresult`](@ref) is the energy, a number. The macro
takes keyword arguments only; `node` is required, the rest optional. It is
[`call_average_energy`](@ref) with `node` given by name. An average energy has no target and no
log scale, so `target` and `logscale` are not accepted.

# Required keywords

$(DOC_CALL_NODE)

# Optional keywords

An average energy reads marginals, `q` and `clusters`, one per cluster of the factorisation.

$(DOC_CALL_INPUTS)

$(DOC_CALL_ALGORITHM)

$(DOC_CALL_CTX)

$(DOC_CALL_ANN)

# The result

A [`RuleResult`](@ref): [`getresult`](@ref) the energy; [`getlogscale`](@ref) is `nothing`;
[`getrule`](@ref) the [`RuleSpec`](@ref) that ran.

$(DOC_CALL_ERRORS)

# Example

```julia
result = @call_average_energy(
    node = NormalMeanVariance,
    q    = (out = NormalMeanVariance(1.0, 1.0), μ = NormalMeanVariance(0.0, 2.0), v = PointMass(1.0)),
)
getresult(result)
```
"""
macro call_average_energy(args...)
    return esc(keyword_call("call_average_energy", call_average_energy, (:node,), args))
end

"""
    @which_message_update_rule(node = ..., target = ..., m = (...), q = (...), clusters = (...), algorithm = ...)

The [`RuleSpec`](@ref) that [`@call_message_update_rule`](@ref) would run for these inputs,
without running it; it displays the rule with its source, file and line. The macro takes
keyword arguments only; `node` and `target` are required, the rest optional. It is
[`which_message_update_rule`](@ref) with `node` and `target` given by name.

# Required keywords

$(DOC_CALL_NODE)

$(DOC_CALL_MESSAGE_TARGET)

# Optional keywords

Only the inputs' types matter, as they are what selects a rule.

$(DOC_CALL_INPUTS)

$(DOC_CALL_ALGORITHM)

When no rule fits, the query throws a [`RuleNotFoundError`](@ref), which lists the closest rules
and why each does not fit.
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
    @which_marginal_update_rule(node = ..., target = (:y, :x), m = (...), q = (...), clusters = (...), algorithm = ...)

The [`RuleSpec`](@ref) that [`@call_marginal_update_rule`](@ref) would run for these inputs,
without running it; it displays the rule with its source, file and line. The macro takes
keyword arguments only; `node` and `target` are required, the rest optional. It is
[`which_marginal_update_rule`](@ref) with `node` and `target` given by name.

# Required keywords

$(DOC_CALL_NODE)

$(DOC_CALL_MARGINAL_TARGET)

# Optional keywords

Only the inputs' types matter, as they are what selects a rule.

$(DOC_CALL_INPUTS)

$(DOC_CALL_ALGORITHM)

When no rule fits, the query throws a [`RuleNotFoundError`](@ref), which lists the closest rules
and why each does not fit.
"""
macro which_marginal_update_rule(args...)
    return esc(keyword_call("which_marginal_update_rule", which_marginal_update_rule, (:node, :target), args))
end

"""
    @which_average_energy(node = ..., q = (...), clusters = (...), m = (...), algorithm = ...)

The [`RuleSpec`](@ref) that [`@call_average_energy`](@ref) would run for these marginals,
without running it; it displays the energy with its source, file and line. The macro takes
keyword arguments only; `node` is required, the rest optional. It is
[`which_average_energy`](@ref) with `node` given by name.

# Required keywords

$(DOC_CALL_NODE)

# Optional keywords

Only the inputs' types matter, as they are what selects the energy.

$(DOC_CALL_INPUTS)

$(DOC_CALL_ALGORITHM)

When no energy fits, the query throws a [`RuleNotFoundError`](@ref), which lists the closest
ones and why each does not fit.
"""
macro which_average_energy(args...)
    return esc(keyword_call("which_average_energy", which_average_energy, (:node,), args))
end
