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

member_label(member::Symbol) = string(member)
member_label((group, k)::Tuple) = "$group[$k]"

function edge_views(r::RuleResult)
    spec, args = r.rule, r.arguments
    targets = target_members(r.target)
    interfaces = applicable(nodespec, spec.node) ? nodespec(spec.node).interfaces : nothing
    edges = EdgeView[]
    if interfaces === nothing
        # A node without a declaration: its edges are what the call names.
        members = unique([targets; [key for key in keys(args.m.values)]; [key for key in keys(args.q.singles)]])
        for member in members
            role, value, detail = edge_role(member, args, targets)
            push!(edges, EdgeView(member_label(member), role, value, detail))
        end
        return edges
    end
    for interface in interfaces
        if !interface.group
            role, value, detail = edge_role(interface.name, args, targets)
            push!(edges, EdgeView(string(interface.name), role, value, detail))
            continue
        end
        n = group_length(interface.name, args, targets)
        if n == 0
            push!(edges, EdgeView("$(interface.name)…", :unused, nothing, ""))
            continue
        end
        for k in 1:n
            role, value, detail = edge_role((interface.name, k), args, targets)
            push!(edges, EdgeView(member_label((interface.name, k)), role, value, detail))
        end
    end
    return edges
end

# A node by its own name, without the module it is defined in: `NormalMeanVariance`, `+`.
node_name(node::Union{Type, Function}) = string(nameof(node))
node_name(node) = string(node)

function call_mode(r::RuleResult)
    r.rule.kind === :average_energy && return "average energy"
    r.rule.kind === :marginal && return "marginal"
    args = r.arguments
    isempty(args.q.singles) && isempty(args.q.joints) && return "belief propagation"
    isempty(args.m.values) && return "variational"
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
    return string(logscale, "  (", logscale_source(r.rule.logscale), ")")
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
    text = replace(strip(sprint(show, value; context)), r"\s*\n\s*" => " ")
    return length(text) > limit ? first(text, limit - 1) * "…" : text
end

# A rule's file as its directory and name, which is what tells rules apart.
short_path(file) = (parts = splitpath(String(file)); joinpath(parts[max(end - 1, 1):end]...))

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

# Distinct ids for the SVG markers of every card on a page.
const HTML_CARD_COUNTER = Ref(0)

const HTML_CARD_STYLE = """
.mprb-card{--mprb-fg:#1f2328;--mprb-muted:#6e7781;--mprb-bg:#ffffff;--mprb-border:#d0d7de;--mprb-target:#1a7f37;--mprb-message:#0969da;--mprb-marginal:#8250df;--mprb-warn:#9a6700;
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:13px;color:var(--mprb-fg);background:var(--mprb-bg);border:1px solid var(--mprb-border);border-radius:8px;padding:12px 14px;margin:6px 0;max-width:960px}
@media (prefers-color-scheme: dark){.mprb-card{--mprb-fg:#e6edf3;--mprb-muted:#8d96a0;--mprb-bg:#0d1117;--mprb-border:#30363d;--mprb-target:#3fb950;--mprb-message:#58a6ff;--mprb-marginal:#bc8cff;--mprb-warn:#d29922}}
.mprb-card .mprb-head{font-weight:600;margin-bottom:8px}.mprb-card .mprb-mode{color:var(--mprb-muted);font-weight:400;margin-left:6px}
.mprb-card .mprb-body{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start}.mprb-card .mprb-sections{flex:1;min-width:280px}
.mprb-card details{margin:4px 0}.mprb-card summary{cursor:pointer;font-weight:600}
.mprb-card table{border-collapse:collapse;margin:4px 0 6px 0;color:inherit;font-size:inherit;font-family:inherit}.mprb-card td,.mprb-card th{padding:2px 8px 2px 0;text-align:left;vertical-align:top}
.mprb-card th{color:var(--mprb-muted);font-weight:500}.mprb-card code,.mprb-card pre{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.mprb-card pre{white-space:pre-wrap;margin:2px 0}.mprb-card .mprb-ok{color:var(--mprb-target)}.mprb-card .mprb-no{color:var(--mprb-warn)}.mprb-card .mprb-undefined{color:var(--mprb-warn)}
.mprb-card svg text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;fill:var(--mprb-fg);stroke:none}
.mprb-card svg text.target{fill:var(--mprb-target)}.mprb-card svg marker path{stroke:none}
.mprb-card svg marker path.target{fill:var(--mprb-target)}.mprb-card svg marker path.message{fill:var(--mprb-message)}
.mprb-card svg marker path.marginal,.mprb-card svg marker path.joint{fill:var(--mprb-marginal)}
.mprb-card svg line.edge{stroke-width:1.6;fill:none}.mprb-card svg line.target{stroke:var(--mprb-target);stroke-width:2.4}
.mprb-card svg line.message{stroke:var(--mprb-message)}.mprb-card svg line.marginal,.mprb-card svg line.joint{stroke:var(--mprb-marginal);stroke-dasharray:5 3}
.mprb-card svg line.unused{stroke:var(--mprb-border);stroke-dasharray:2 3}.mprb-card svg text.unused{fill:var(--mprb-muted)}
.mprb-card svg text.message{fill:var(--mprb-message)}.mprb-card svg text.marginal,.mprb-card svg text.joint{fill:var(--mprb-marginal)}
.mprb-card svg .node{fill:var(--mprb-bg);stroke:var(--mprb-fg);stroke-width:1.6}
"""

role_class(role) = string(role)

# The node as a box with its edges: the target(s) on the right with arrows out, every other edge
# on the left, an arrow into the box for each input, greyed where unused.
function html_node_svg(io::IO, r::RuleResult, edges, id)
    left = [e for e in edges if e.role !== :target]
    right = [e for e in edges if e.role === :target]
    rows = max(length(left), length(right), 1)
    height = 30 * rows + 30
    label_width = 8 * maximum(e -> length(e.label), edges; init = 3) + 10
    name = node_name(r.rule.node)
    box_width = max(60, 8 * length(name) + 16)
    width = 2 * label_width + 2 * 70 + box_width
    box_x, box_y, box_h = label_width + 70, 15, height - 30
    print(io, "<svg class=\"mprb-node\" role=\"img\" aria-label=\"", html_escape(result_label(r)), "\" width=\"", width, "\" height=\"", height, "\" viewBox=\"0 0 ", width, " ", height, "\">")
    print(io, "<defs>")
    for role in (:target, :message, :marginal, :joint)
        print(io, "<marker id=\"", id, "-", role, "\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" markerUnits=\"userSpaceOnUse\" markerWidth=\"9\" markerHeight=\"9\" orient=\"auto-start-reverse\">")
        print(io, "<path d=\"M0,0 L10,5 L0,10 z\" class=\"", role, "\"/></marker>")
    end
    print(io, "</defs>")
    y_of(i, n) = box_y + box_h * i / (n + 1)
    for (i, edge) in enumerate(left)
        y = y_of(i, length(left))
        marker = edge.role === :unused ? "" : " marker-end=\"url(#$id-$(edge.role))\""
        print(io, "<line class=\"edge ", role_class(edge.role), "\" x1=\"", label_width, "\" y1=\"", y, "\" x2=\"", box_x, "\" y2=\"", y, "\"", marker, "/>")
        print(io, "<text class=\"", role_class(edge.role), "\" x=\"", label_width - 6, "\" y=\"", y + 4, "\" text-anchor=\"end\">", html_escape(edge.label), "</text>")
    end
    for (i, edge) in enumerate(right)
        y = y_of(i, length(right))
        x1, x2 = box_x + box_width, box_x + box_width + 70
        print(io, "<line class=\"edge target\" x1=\"", x1, "\" y1=\"", y, "\" x2=\"", x2, "\" y2=\"", y, "\" marker-end=\"url(#", id, "-target)\"/>")
        print(io, "<text class=\"target\" x=\"", x2 + 8, "\" y=\"", y + 4, "\" font-weight=\"bold\">", html_escape(edge.label), "</text>")
    end
    print(io, "<rect class=\"node\" x=\"", box_x, "\" y=\"", box_y, "\" width=\"", box_width, "\" height=\"", box_h, "\" rx=\"4\"/>")
    print(io, "<text x=\"", box_x + box_width / 2, "\" y=\"", box_y + box_h / 2 + 4, "\" text-anchor=\"middle\">", html_escape(name), "</text>")
    print(io, "</svg>")
    return nothing
end

function html_rows(io::IO, rows)
    print(io, "<table>")
    for (key, value) in rows
        print(io, "<tr><th>", html_escape(key), "</th><td>", value, "</td></tr>")
    end
    print(io, "</table>")
    return nothing
end

html_code(value) = "<code>" * html_escape(value) * "</code>"

function Base.show(io::IO, ::MIME"text/html", r::RuleResult)
    id = "mprb-" * string(HTML_CARD_COUNTER[] += 1)
    edges = edge_views(r)
    print(io, "<div class=\"mprb-card\" id=\"", id, "\"><style>", HTML_CARD_STYLE, "</style>")
    print(io, "<div class=\"mprb-head\">RuleResult: ", html_escape(result_label(r)), "<span class=\"mprb-mode\">", html_escape(call_mode(r)), "</span></div>")
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
