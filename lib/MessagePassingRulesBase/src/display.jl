# REPL (`text/plain`) and notebook (`text/html`) display of targets and specs.

Base.show(io::IO, ::Target{E}) where {E} = print(io, repr(E))
Base.show(io::IO, target::IndexedTarget{E}) where {E} = print(io, "(", repr(E), ", ", target.index, ")")
Base.show(io::IO, ::ClusterTarget{K}) where {K} = print(io, repr(K))

yesno(flag) = flag ? "yes" : "no"
kind_label(kind) = kind === :average_energy ? "average energy" : "$kind rule"

# A value as `io` would print it: a type defined in a module other than the one `io` shows from
# is qualified only as far as that module requires, as at the REPL. Without `io`, as `string`.
shown(io, x) = prettify_modules(io === nothing ? string(x) : sprint(print, x; context = io))

# A module Julia generates, rather than one a package defines: Documenter's sandboxes,
# `Main.var"__atexample__named__first-node"`, a test item's, a `gensym`'d one. Such a module means
# nothing to a reader, so a printed name drops it; a package's module, `ExponentialFamily.`,
# stays. A generated name is quoted with `var"…"`, starts with `__`, or contains a `#`.
const GENERATED_MODULE = r"(?:\bMain\.)?(?:var\"[^\"]*\"|__[A-Za-z0-9_]*|[A-Za-z0-9_]*#[A-Za-z0-9_#]*)\.(?=[A-Za-z_]|var\")"

"""
    prettify_modules(text::AbstractString) -> String

`text`, a printed name, type or value, without the modules Julia generates: Documenter's
sandboxes (`Main.var"__atexample__named__page".Gaussian` becomes `Gaussian`), test items' and
`gensym`'d modules. A package's module stays. The displays of this package print through it.
"""
prettify_modules(text::AbstractString) = replace(text, GENERATED_MODULE => "")

function rule_heading(spec::RuleSpec, io = nothing)
    heading = "$(kind_label(spec.kind)) for $(node_name(spec.node))"
    spec.kind === :average_energy || (heading *= " towards $(target_label(spec.target))")
    return heading * " under $(shown(io, spec.algorithm))"
end

function inputs_label(spec::RuleSpec, io = nothing)
    labels = [spec.default ? ["default"] : String[]; [input_label(i.container, i.key, i.selection) * "::" * shown(io, i.type) for i in spec.inputs]]
    return isempty(labels) ? "none" : join(labels, ", ")
end

Base.show(io::IO, input::InputSpec) = print(io, input_label(input.container, input.key, input.selection), "::", shown(io, input.type))

function Base.show(io::IO, r::RuleNotFound)
    print(io, "RuleNotFound(", kind_label(r.kind), " for ", shown(io, r.node))
    r.target === nothing || print(io, " towards ", r.target)
    return print(io, " under ", shown(io, r.algorithm), ")")
end

Base.show(io::IO, spec::RuleSpec) = print(io, "RuleSpec(", rule_heading(spec, io), " @ ", short_path(spec.file), ":", spec.line, ")")

function Base.show(io::IO, ::MIME"text/plain", spec::RuleSpec)
    println(io, "RuleSpec: ", rule_heading(spec, io))
    println(io, "  inputs:   ", inputs_label(spec, io))
    println(io, "  in-place: ", yesno(spec.inplace), " · scratch: ", yesno(spec.scratch !== nothing), " · pure: ", yesno(spec.pure), " · services: ", isempty(spec.services) ? "none" : join(spec.services, ", "))
    spec.kind === :message && println(io, "  logscale: ", describe_logscale_declaration(spec.logscale), spec.reads_logscale ? " · reads incoming log scales" : "")
    spec.args_check === nothing || println(io, "  checks:   ", spec.args_check)
    println(io, "  defined:  ", short_path(spec.file), ":", spec.line)
    print(io, "  body:     ", spec.source)
    return nothing
end

# The role an input plays in a drawing: a joint's members, a marginal or a message.
input_role(container, key) = key isa Tuple ? :joint : container === :m ? :message : :marginal

function rule_edges(spec::RuleSpec)
    left = [spec.default ? [SvgEdge("default", :default)] : SvgEdge[]; [SvgEdge(input_label(i.container, i.key, i.selection), input_role(i.container, i.key)) for i in spec.inputs]]
    spec.kind === :average_energy && return left, SvgEdge[]
    target = spec.target
    label = target === ClusterTarget ? "q(any cluster)" :
        target <: ClusterTarget ? "q(" * join(map(member_label, target.parameters[1]), ", ") * ")" :
        target <: IndexedTarget ? "$(target.parameters[1])[k]" : string(target.parameters[1])
    return left, [SvgEdge(label, :target)]
end

function drawing(spec::RuleSpec)
    left, right = rule_edges(spec)
    part = SvgPart(node_name(spec.node), left, right; aria = rule_heading(spec))
    return svg_drawing([part], "RuleSpec: " * rule_heading(spec); roles = Set(e.role for e in [left; right]))
end

function Base.show(io::IO, ::MIME"text/html", spec::RuleSpec)
    id = open_card(io, "RuleSpec: " * rule_heading(spec, io))
    left, right = rule_edges(spec)
    print(io, "<div class=\"mprb-body\"><figure class=\"mprb-figure\">")
    svg_node(io, id, node_name(spec.node); left, right, aria = rule_heading(spec, io))
    html_legend(io, Set(e.role for e in [left; right]))
    print(io, "</figure><div class=\"mprb-sections\">")
    rows = Pair{String, String}[
        "inputs" => html_code(inputs_label(spec, io)),
        "in-place" => yesno(spec.inplace), "scratch" => yesno(spec.scratch !== nothing), "pure" => yesno(spec.pure),
        "services" => isempty(spec.services) ? "none" : html_code(join(spec.services, ", ")),
    ]
    spec.kind === :message && push!(rows, "log scale" => html_escape(describe_logscale_declaration(spec.logscale)) * (spec.reads_logscale ? ", reads the incoming ones" : ""))
    spec.args_check === nothing || push!(rows, "checks" => html_code(spec.args_check))
    push!(rows, "defined" => html_code("$(short_path(spec.file)):$(spec.line)"))
    push!(rows, "body" => "<pre>" * html_escape(spec.source) * "</pre>")
    html_rows(io, rows)
    print(io, "</div></div>")
    close_card(io)
    return nothing
end

function interface_label(interface::InterfaceSpec)
    label = string(interface.name) * (interface.group ? "..." : "")
    isempty(interface.aliases) || (label *= " (aliases: " * join(interface.aliases, ", ") * ")")
    return label
end

type_label(spec::NodeSpec) = spec.type isa Stochastic ? "stochastic" : "deterministic"

Base.show(io::IO, spec::NodeSpec) = print(io, "NodeSpec(", shown(io, spec.node), ", ", type_label(spec), ")")

function Base.show(io::IO, ::MIME"text/plain", spec::NodeSpec)
    println(io, "NodeSpec: ", shown(io, spec.node), " (", type_label(spec), ")")
    println(io, "  interfaces:        ", join(map(interface_label, spec.interfaces), ", "))
    println(io, "  default algorithm: ", shown(io, spec.algorithm))
    println(io, "  static inputs:     ", spec.static_inputs)
    isempty(spec.matched_groups) || println(io, "  matched groups:    ", join(map(g -> join(g, " = "), spec.matched_groups), ", "))
    spec.min_group_length == 1 || println(io, "  min group length:  ", spec.min_group_length)
    spec.factorisation === :any || println(io, "  factorisation:     ", spec.factorisation)
    isempty(spec.initial_messages) || println(io, "  initial messages:  ", join(map(p -> "$(first(p)) => $(last(p))", spec.initial_messages), ", "))
    print(io, "  defined:           ", short_path(spec.file), ":", spec.line)
    return nothing
end

Base.show(io::IO, interface::InterfaceSpec) = print(io, interface_label(interface))

interface_edge(interface::InterfaceSpec) =
    SvgEdge(string(interface.name) * (interface.group ? "…" : ""), :interface, join(interface.aliases, ", "))

# The output on the right, every other interface on the left.
function node_part(spec::NodeSpec)
    edges = map(interface_edge, collect(spec.interfaces))
    return SvgPart(node_name(spec.node), edges[2:end], edges[1:min(1, end)]; kind = type_label(spec), aria = "the node $(node_name(spec.node)) and its interfaces")
end

drawing(spec::NodeSpec) = svg_drawing([node_part(spec)], "NodeSpec: " * node_name(spec.node) * " (" * type_label(spec) * ")"; aria = node_part(spec).aria)

function Base.show(io::IO, ::MIME"text/html", spec::NodeSpec)
    id = open_card(io, "NodeSpec: " * node_name(spec.node))
    part = node_part(spec)
    print(io, "<div class=\"mprb-body\"><figure class=\"mprb-figure\">")
    svg_node(io, id, part.name; part.left, part.right, part.kind, part.aria)
    print(io, "</figure><div class=\"mprb-sections\">")
    print(io, "<table><tr><th>interface</th><th></th><th>aliases</th></tr>")
    for interface in spec.interfaces
        print(io, "<tr><td>", html_code(interface.name), "</td><td>", interface.group ? "a group of any number of members" : "", "</td><td>", join(map(html_code, interface.aliases), ", "), "</td></tr>")
    end
    print(io, "</table>")
    rows = Pair{String, String}["default algorithm" => html_code(shown(io, spec.algorithm)), "static inputs" => html_code(spec.static_inputs)]
    isempty(spec.matched_groups) || push!(rows, "matched groups" => html_code(join(map(g -> join(g, " = "), spec.matched_groups), ", ")))
    spec.min_group_length == 1 || push!(rows, "min group length" => string(spec.min_group_length))
    spec.factorisation === :any || push!(rows, "factorisation" => html_code(spec.factorisation))
    isempty(spec.initial_messages) || push!(rows, "initial messages" => html_code(join(map(p -> "$(first(p)) => $(shown(io, last(p)))", spec.initial_messages), ", ")))
    push!(rows, "defined" => html_code("$(short_path(spec.file)):$(spec.line)"))
    html_rows(io, rows)
    print(io, "</div></div>")
    close_card(io)
    return nothing
end

# A dependency as a declaration writes it. `dependency_label` is the form a rule's input matches,
# where a custom selection of a group's members is the whole group, `m[:x...]`.
dependency_display(d::Dependency) = d.selector isa CustomGroupSelector ?
    "$(d.container)[:$(d.key)][select_group_members(…; arity = $(d.selector.arity))]" : dependency_label(d)

dependency_target_label(entry::TargetDependencies) = entry.indexed ? "(:$(entry.edge), k)" : ":$(entry.edge)"
function dependency_inputs_label(entry::TargetDependencies)
    labels = [entry.default ? ["default"] : String[]; map(dependency_display, collect(entry.inputs))]
    return isempty(labels) ? "nothing" : join(labels, ", ")
end
partition_label(spec::DependenciesSpec) =
    spec.partition === nothing ? "from the factorisation" : join(map(repr, spec.partition), ", ")

Base.show(io::IO, spec::DependenciesSpec) = print(io, "DependenciesSpec(", shown(io, spec.node), ", ", shown(io, spec.algorithm), ")")

function Base.show(io::IO, ::MIME"text/plain", spec::DependenciesSpec)
    println(io, "DependenciesSpec: ", shown(io, spec.node), " under ", shown(io, spec.algorithm))
    width = maximum(length ∘ dependency_target_label, spec.targets; init = 0)
    for entry in spec.targets
        println(io, "  ", rpad(dependency_target_label(entry), width), " ⇐ ", dependency_inputs_label(entry))
    end
    print(io, "  free-energy partition: ", partition_label(spec))
    return nothing
end

Base.show(io::IO, d::Dependency) = print(io, dependency_display(d))
Base.show(io::IO, entry::TargetDependencies) = print(io, dependency_target_label(entry), " ⇐ ", dependency_inputs_label(entry))

dependency_edges(entry::TargetDependencies) =
    [entry.default ? [SvgEdge("default", :default)] : SvgEdge[]; [SvgEdge(dependency_display(d), input_role(d.container, d.key)) for d in entry.inputs]]

# One small node per target, with the inputs its rules take.
function dependency_part(spec::DependenciesSpec, entry::TargetDependencies)
    target = entry.indexed ? "$(entry.edge)[k]" : string(entry.edge)
    return SvgPart(node_name(spec.node), dependency_edges(entry), [SvgEdge(target, :target)]; aria = "the inputs of the rules towards $target", stub = 40)
end
dependency_roles(parts) = union(Set([:target]), (e.role for p in parts for e in p.left))

function drawing(spec::DependenciesSpec)
    parts = [dependency_part(spec, entry) for entry in spec.targets]
    description = "DependenciesSpec: $(node_name(spec.node)) under $(shown(nothing, spec.algorithm))"
    return svg_drawing(parts, description; roles = dependency_roles(parts))
end

function Base.show(io::IO, ::MIME"text/html", spec::DependenciesSpec)
    id = open_card(io, "DependenciesSpec: $(node_name(spec.node))", "under $(shown(io, spec.algorithm))")
    print(io, "<div class=\"mprb-grid\">")
    parts = [dependency_part(spec, entry) for entry in spec.targets]
    for part in parts
        svg_node(io, id, part.name; part.left, part.right, part.aria, part.stub)
    end
    roles = dependency_roles(parts)
    print(io, "</div>")
    html_legend(io, roles)
    html_rows(io, ["free-energy partition" => html_escape(partition_label(spec))])
    close_card(io)
    return nothing
end

Base.show(io::IO, coverage::RuleCoverage) = print(io, "RuleCoverage(", shown(io, coverage.node), ")")

coverage_cell(coverage, row, algorithm) =
    (n = get(coverage.counts, (row, algorithm), 0); n == 0 ? "" : n == 1 ? "✓" : "✓×$n")

# A column of a coverage table: the algorithm's name with its parameters, unqualified, so that the
# variants of a parametric algorithm, `BinomialPolyaApproximation{Int64}` and the rules declared on
# `BinomialPolyaApproximation` itself, are told apart.
# A variant bounded by its parameters, `FlowApproximation{<:AbstractCompiledFlowModel, <:Linearization}`,
# keeps its bounds, `_` for a free one; with every parameter free it is the bare name.
function algorithm_label(algorithm::UnionAll)
    body = Base.unwrap_unionall(algorithm)
    parameters = body.parameters
    all(p -> p isa TypeVar && p.ub === Any, parameters) && return string(nameof(body))
    return string(nameof(body), "{", join(map(bound_label, parameters), ", "), "}")
end
bound_label(p::TypeVar) = p.ub === Any ? "_" : "<:" * algorithm_label(p.ub)
bound_label(p) = parameter_label(p)
function algorithm_label(algorithm::DataType)
    parameters = algorithm.parameters
    isempty(parameters) && return string(nameof(algorithm))
    return string(nameof(algorithm), "{", join(map(parameter_label, parameters), ", "), "}")
end
algorithm_label(algorithm) = string(algorithm)
# A parameter that is itself a type by its bare name, so a column reads
# `GCVApproximation{GaussHermiteCubature}`, not the cubature's own parameters.
parameter_label(parameter::Union{DataType, UnionAll}) = string(nameof(Base.unwrap_unionall(parameter)))
parameter_label(parameter::Type) = string(parameter)
parameter_label(parameter) = repr(parameter)

function Base.show(io::IO, ::MIME"text/plain", coverage::RuleCoverage)
    println(io, "Rule coverage for ", shown(io, coverage.node))
    names = map(algorithm_label, coverage.algorithms)
    rowwidth = maximum(length, coverage.rows; init = 0)
    widths = map(name -> max(length(name), 3), names)
    print(io, "  ", " "^rowwidth)
    foreach((name, w) -> print(io, " │ ", rpad(name, w)), names, widths)
    for row in coverage.rows
        print(io, "\n  ", rpad(row, rowwidth))
        foreach((algorithm, w) -> print(io, " │ ", rpad(coverage_cell(coverage, row, algorithm), w)), coverage.algorithms, widths)
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/html", coverage::RuleCoverage)
    print(io, "<table><caption>Rule coverage for ", html_escape(shown(io, coverage.node)), "</caption><tr><th></th>")
    foreach(a -> print(io, "<th>", html_escape(algorithm_label(a)), "</th>"), coverage.algorithms)
    print(io, "</tr>")
    for row in coverage.rows
        print(io, "<tr><th style=\"text-align:left\">", html_escape(row), "</th>")
        foreach(a -> print(io, "<td>", coverage_cell(coverage, row, a), "</td>"), coverage.algorithms)
        print(io, "</tr>")
    end
    print(io, "</table>")
    return nothing
end
