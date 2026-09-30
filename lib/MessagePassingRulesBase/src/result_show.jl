# The rich display of a `RuleResult`: a report in the terminal (`text/plain`) and a card with the
# node drawn in notebooks and documentation (`text/html`). Both render one view of the result:
# the node's interfaces and what each carried into the rule, the result and its log scale, what
# ran, and the other rules for the same target.

# One edge of the node as the call saw it: `role` is `:target`, `:message`, `:marginal`, `:joint`
# (a member of a joint marginal, `detail` its key) or `:unused`.
struct EdgeView
    label::String
    role::Symbol
    value::Any
    detail::String
end

target_members(target::Target{E}) where {E} = Any[E]
target_members(target::IndexedTarget{E}) where {E} = Any[(E, target.index)]
target_members(::ClusterTarget{K}) where {K} = collect(Any, K)
target_members(_) = Any[]

joint_label(key) = input_label(:q, key, :cluster)

# Whether the joint `key` covers member `member` (an interface name, or `(group, k)`): a group's
# name in a joint stands for all its members.
covers(key, member::Symbol) = member in key
covers(key, member::Tuple) = member in key || first(member) in key

function edge_role(member, args::RuleArgs, targets)
    member in targets && return :target, nothing, ""
    name, index = member isa Tuple ? member : (member, nothing)
    for (container, role) in ((args.m.values, :message), (args.q.singles, :marginal))
        haskey(container, name) || continue
        value = getfield(container, name)
        index === nothing && return role, value, ""
        value isa Tuple && index <= length(value) && value[index] !== nothing && return role, value[index], ""
    end
    for (key, value) in zip(joint_keys(args.q), args.q.joints)
        covers(key, member) && return :joint, value, joint_label(key)
    end
    return :unused, nothing, ""
end

# The members a group had in this call: as many as its tuple in the arguments, or as the largest
# index a target or a joint names. `0` when the call says nothing about it.
function group_length(name, args::RuleArgs, targets)
    n = 0
    for container in (args.m.values, args.q.singles)
        haskey(container, name) && (n = max(n, length(getfield(container, name))))
    end
    for member in [targets; collect(Iterators.flatten(joint_keys(args.q)))]
        member isa Tuple && first(member) === name && (n = max(n, last(member)))
    end
    return n
end


# An edge, and when it is a target the rule also reads, a second edge for what it read there: a
# rule towards `in` may take the message on `in` itself, which the target's arrow would hide.
function push_edge!(edges, label, member, args, targets)
    role, value, detail = edge_role(member, args, targets)
    push!(edges, EdgeView(label, role, value, detail))
    if role === :target
        role, value, detail = edge_role(member, args, ())
        role === :unused || push!(edges, EdgeView(label, role, value, isempty(detail) ? "its own edge" : detail * ", its own edge"))
    end
    return edges
end

function edge_views(r::RuleResult)
    # A marginal rule reads the messages on its cluster's members and computes their joint: the
    # members are drawn as inputs, and the cluster as the one edge out.
    if r.rule.kind === :marginal
        members = target_members(r.target)
        edges = member_edges(r, Any[])
        push!(edges, EdgeView("q(" * join(map(member_label, members), ", ") * ")", :target, nothing, ""))
        return edges
    end
    return member_edges(r, target_members(r.target))
end

function member_edges(r::RuleResult, targets)
    spec, args = r.rule, r.arguments
    interfaces = applicable(nodespec, spec.node) ? nodespec(spec.node).interfaces : nothing
    edges = EdgeView[]
    if interfaces === nothing
        # A node without a declaration: its edges are what the call names.
        members = unique([targets; [key for key in keys(args.m.values)]; [key for key in keys(args.q.singles)]])
        foreach(member -> push_edge!(edges, member_label(member), member, args, targets), members)
        return edges
    end
    for interface in interfaces
        if !interface.group
            push_edge!(edges, string(interface.name), interface.name, args, targets)
            continue
        end
        n = group_length(interface.name, args, targets)
        if n == 0
            # A joint over the whole group, `q[(:in,)]`, says nothing of its members' number.
            position = findfirst(key -> interface.name in key, collect(joint_keys(args.q)))
            if position === nothing
                push!(edges, EdgeView("$(interface.name)…", :unused, nothing, ""))
            else
                key = joint_keys(args.q)[position]
                push!(edges, EdgeView("$(interface.name)…", :joint, args.q.joints[position], joint_label(key)))
            end
            continue
        end
        foreach(k -> push_edge!(edges, member_label((interface.name, k)), (interface.name, k), args, targets), 1:n)
    end
    return edges
end

function call_mode(r::RuleResult)
    r.rule.kind === :average_energy && return "average energy"
    r.rule.kind === :marginal && return "marginal"
    args = r.arguments
    messages_only, marginals_only = isempty(args.q.singles) && isempty(args.q.joints), isempty(args.m.values)
    # Under the default algorithm the inputs say which scheme ran; another algorithm, an
    # expectation propagation rule say, is named by its inputs alone.
    if r.algorithm isa DefaultAlgorithm
        messages_only && return "belief propagation"
        marginals_only && return "variational"
        return "messages and marginals"
    end
    messages_only && return "messages"
    marginals_only && return "marginals"
    return "messages and marginals"
end

function result_label(r::RuleResult)
    spec = r.rule
    spec.kind === :average_energy && return "average energy of $(node_name(spec.node))"
    spec.kind === :marginal && return "marginal of $(node_name(spec.node)) over $(r.target)"
    return "message of $(node_name(spec.node)) towards $(r.target)"
end

logscale_source(::Real) = "declared"
logscale_source(::Function) = "computed from the inputs"
logscale_source(::FromBody) = "computed by the body"
logscale_source(::Nothing) = "not declared"

function logscale_label(r::RuleResult)
    logscale = r.logscale
    logscale isa UndefinedLogScale && return "undefined: " * sprint(describe_undefined, logscale)
    # An irrational shows its value beside its name, `loghalf = -0.6931471805599...`.
    value = logscale isa AbstractIrrational ? repr(MIME"text/plain"(), logscale) : string(logscale)
    return string(value, "  (", logscale_source(r.rule.logscale), ")")
end

input_labels(m::Messages) = Pair{String, Any}["m[$(repr(key))]" => value for (key, value) in pairs(m.values)]
input_labels(q::Marginals{N, T, J}) where {N, T, J} = Pair{String, Any}[
    ["q[$(repr(key))]" => value for (key, value) in pairs(q.singles)];
    ["q[$(repr(key))]" => value for (key, value) in zip(J, q.joints)]
]

# The other rules for the same node, target and kind, each with how it fits this call.
function other_rules(r::RuleResult)
    spec = r.rule
    target = spec.kind === :average_energy ? nothing : r.target
    candidates = candidate_rules(RuleNotFound(spec.kind, spec.node, target, r.algorithm, r.arguments))
    provided = provided_inputs(r.arguments)
    return [(candidate, fit_report(candidate, r.algorithm, provided)) for candidate in candidates if candidate !== spec]
end

# One line, however the value shows itself.
function compact_repr(value; limit = 120, io = nothing)
    context = io === nothing ? (:compact => true, :limit => true) : IOContext(io, :compact => true, :limit => true)
    text = prettify_modules(replace(strip(sprint(show, value; context)), r"\s*\n\s*" => " "))
    return length(text) > limit ? first(text, limit - 1) * "…" : text
end


# A joint arrives once; its members after the first only name it.
function first_of_joints(edges)
    seen = Set{String}()
    return map(edges) do edge
        edge.role === :joint || return true
        edge.detail in seen && return false
        push!(seen, edge.detail)
        return true
    end
end

## text/plain

const ROLE_COLORS = Dict(:target => :green, :message => :cyan, :marginal => :magenta, :joint => :magenta, :unused => :light_black)
const ROLE_ARROWS = Dict(:target => "◀══", :message => "──▶", :marginal => "┄┄▶", :joint => "┄┄▶", :unused => "   ")
const ROLE_TAGS = Dict(:target => "target", :message => "m", :marginal => "q", :joint => "q", :unused => "unused")

function Base.show(io::IO, ::MIME"text/plain", r::RuleResult)
    get(io, :compact, false) && return show(io, r)
    printstyled(io, "RuleResult"; bold = true)
    print(io, "  ", result_label(r), "  ")
    printstyled(io, "· ", call_mode(r); color = :light_black)
    edges = edge_views(r)
    width = maximum(e -> length(e.label), edges; init = 0)
    print(io, "\n  ")
    printstyled(io, node_name(r.rule.node); bold = true)
    for (edge, first) in zip(edges, first_of_joints(edges))
        print(io, "\n    ", rpad(edge.label, width), "  ")
        color = ROLE_COLORS[edge.role]
        printstyled(io, ROLE_ARROWS[edge.role], "  ", ROLE_TAGS[edge.role]; color, bold = edge.role === :target)
        if edge.role === :target
            print(io, "  ")
            printstyled(io, compact_repr(r.result; io); color = :green)
        elseif edge.role !== :unused
            first && print(io, "  ", compact_repr(edge.value; io))
            isempty(edge.detail) || printstyled(io, "  (", first ? "" : "in ", edge.detail, ")"; color = :light_black)
        end
    end
    print(io, "\n  result     ", compact_repr(r.result; io))
    if r.logscale !== nothing
        print(io, "\n  logscale   ")
        printstyled(io, logscale_label(r); color = r.logscale isa UndefinedLogScale ? :yellow : :normal)
    end
    args = r.arguments
    if args.logscale !== nothing
        print(io, "\n  incoming   ", join(("m[$(repr(k))] = $(compact_repr(v; io))" for (k, v) in pairs(args.logscale.m.values)), ", "))
    end
    print(io, "\n  algorithm  ", compact_repr(r.algorithm; io))
    isempty(r.rule.services) || print(io, "\n  services   ", join(r.rule.services, ", "))
    r.scratch === nothing || print(io, "\n  scratch    ", compact_repr(r.scratch; io))
    print(io, "\n  rule       ", inputs_label(r.rule), "  @ ", short_path(r.rule.file), ":", r.rule.line)
    others = other_rules(r)
    isempty(others) || printstyled(io, "\n  and ", length(others), " other rule", length(others) == 1 ? "" : "s", " for this target"; color = :light_black)
    return nothing
end

## text/html


# The node as a box with its edges: the target(s) on the right with arrows out, every other edge
# on the left, an arrow into the box for each input, greyed where unused.
function html_node_svg(io::IO, r::RuleResult, edges, id)
    left = [SvgEdge(e.label, e.role) for e in edges if e.role !== :target]
    right = [SvgEdge(e.label, e.role) for e in edges if e.role === :target]
    svg_node(io, id, node_name(r.rule.node); left, right, aria = result_label(r), stub = 70)
    return nothing
end

function Base.show(io::IO, ::MIME"text/html", r::RuleResult)
    edges = edge_views(r)
    id = open_card(io, "RuleResult: " * result_label(r), call_mode(r))
    print(io, "<div class=\"mprb-body\">")
    html_node_svg(io, r, edges, id)
    print(io, "<div class=\"mprb-sections\">")

    print(io, "<details open class=\"mprb-result\"><summary>Result</summary>")
    rows = Pair{String, String}["value" => html_code(compact_repr(r.result; limit = 400, io)), "type" => html_code(shown(io, typeof(r.result)))]
    if r.logscale !== nothing
        push!(rows, "log scale" => (r.logscale isa UndefinedLogScale ? "<span class=\"mprb-undefined\">" * html_escape(logscale_label(r)) * "</span>" : html_escape(logscale_label(r))))
    end
    html_rows(io, rows)
    print(io, "</details>")

    print(io, "<details open class=\"mprb-inputs\"><summary>Inputs</summary><table><tr><th>edge</th><th></th><th>value</th></tr>")
    for (edge, first) in zip(edges, first_of_joints(edges))
        edge.role === :target && continue
        value = edge.role === :unused ? "<span class=\"mprb-no\">unused</span>" : first ? html_code(compact_repr(edge.value; io)) : ""
        detail = isempty(edge.detail) ? "" : " <span class=\"mprb-mode\">" * html_escape(edge.detail) * "</span>"
        print(io, "<tr class=\"", role_class(edge.role), "\"><td>", html_escape(edge.label), "</td><td>", html_escape(ROLE_TAGS[edge.role]), "</td><td>", value, detail, "</td></tr>")
    end
    print(io, "</table>")
    args = r.arguments
    if args.logscale !== nothing
        html_rows(io, ["incoming log scale $(k)" => html_code(compact_repr(v; io)) for (k, v) in pairs(args.logscale.m.values)])
    end
    print(io, "</details>")

    spec = r.rule
    print(io, "<details class=\"mprb-rule\"><summary>Rule</summary>")
    rule_rows = Pair{String, String}[
        "declared inputs" => html_code(inputs_label(spec)),
        "algorithm" => html_code(compact_repr(r.algorithm; io)),
    ]
    isempty(spec.services) || push!(rule_rows, "services" => html_code(join(spec.services, ", ")))
    r.scratch === nothing || push!(rule_rows, "scratch" => html_code(compact_repr(r.scratch; io)))
    spec.kind === :message && push!(rule_rows, "log scale" => html_escape(describe_logscale_declaration(spec.logscale)) * (spec.reads_logscale ? ", reads the incoming ones" : ""))
    push!(rule_rows, "defined" => html_code("$(short_path(spec.file)):$(spec.line)"))
    push!(rule_rows, "body" => "<pre>" * html_escape(spec.source) * "</pre>")
    html_rows(io, rule_rows)
    print(io, "</details>")

    others = other_rules(r)
    if !isempty(others)
        print(io, "<details class=\"mprb-others\"><summary>Other rules for this target (", length(others), ")</summary>")
        for (candidate, lines) in others
            print(io, "<div>", html_code("$(short_path(candidate.file)):$(candidate.line)"), "<table>")
            for (ok, text) in lines
                print(io, "<tr><td class=\"", ok ? "mprb-ok" : "mprb-no", "\">", ok ? "✓" : "✗", "</td><td>", html_code(text), "</td></tr>")
            end
            print(io, "</table></div>")
        end
        print(io, "</details>")
    end
    print(io, "</div></div></div>")
    return nothing
end
