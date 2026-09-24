# REPL (`text/plain`) and notebook (`text/html`) display of targets and specs.

Base.show(io::IO, ::Target{E}) where {E} = print(io, repr(E))
Base.show(io::IO, target::IndexedTarget{E}) where {E} = print(io, "(", repr(E), ", ", target.index, ")")
Base.show(io::IO, ::ClusterTarget{K}) where {K} = print(io, repr(K))

html_escape(x) = replace(string(x), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;", "\"" => "&quot;")

function html_table(io::IO, title, rows)
    print(io, "<table><caption>", html_escape(title), "</caption>")
    for (key, value) in rows
        print(io, "<tr><th style=\"text-align:left\">", html_escape(key), "</th><td><code>", html_escape(value), "</code></td></tr>")
    end
    print(io, "</table>")
    return nothing
end

yesno(flag) = flag ? "yes" : "no"
kind_label(kind) = kind === :average_energy ? "average energy" : "$kind rule"

function rule_heading(spec::RuleSpec)
    heading = "$(kind_label(spec.kind)) for $(spec.node)"
    spec.kind === :average_energy || (heading *= " towards $(target_label(spec.target))")
    return heading * " under $(spec.algorithm)"
end

function inputs_label(spec::RuleSpec)
    labels = [spec.default ? ["default"] : String[]; [input_label(i.container, i.key, i.selection) * "::" * string(i.type) for i in spec.inputs]]
    return isempty(labels) ? "none" : join(labels, ", ")
end

Base.show(io::IO, spec::RuleSpec) = print(io, "RuleSpec(", rule_heading(spec), " @ ", spec.file, ":", spec.line, ")")

function Base.show(io::IO, ::MIME"text/plain", spec::RuleSpec)
    println(io, "RuleSpec: ", rule_heading(spec))
    println(io, "  inputs:   ", inputs_label(spec))
    println(io, "  in-place: ", yesno(spec.inplace), " · scratch: ", yesno(spec.scratch !== nothing), " · pure: ", yesno(spec.pure), " · services: ", isempty(spec.services) ? "none" : join(spec.services, ", "))
    println(io, "  defined:  ", spec.file, ":", spec.line)
    print(io, "  body:     ", spec.source)
    return nothing
end

Base.show(io::IO, ::MIME"text/html", spec::RuleSpec) = html_table(
    io, "RuleSpec: " * rule_heading(spec), [
        "inputs" => inputs_label(spec), "in-place" => yesno(spec.inplace), "scratch" => yesno(spec.scratch !== nothing), "pure" => yesno(spec.pure),
        "services" => isempty(spec.services) ? "none" : join(spec.services, ", "),
        "defined" => "$(spec.file):$(spec.line)", "body" => spec.source,
    ]
)

function interface_label(interface::InterfaceSpec)
    label = string(interface.name) * (interface.group ? "..." : "")
    isempty(interface.aliases) || (label *= " (aliases: " * join(interface.aliases, ", ") * ")")
    return label
end

type_label(spec::NodeSpec) = spec.type isa Stochastic ? "stochastic" : "deterministic"

Base.show(io::IO, spec::NodeSpec) = print(io, "NodeSpec(", spec.node, ", ", type_label(spec), ")")

function Base.show(io::IO, ::MIME"text/plain", spec::NodeSpec)
    println(io, "NodeSpec: ", spec.node, " (", type_label(spec), ")")
    println(io, "  interfaces:        ", join(map(interface_label, spec.interfaces), ", "))
    println(io, "  default algorithm: ", spec.algorithm)
    println(io, "  static inputs:     ", spec.static_inputs)
    isempty(spec.matched_groups) || println(io, "  matched groups:    ", join(map(g -> join(g, " = "), spec.matched_groups), ", "))
    spec.min_group_length == 1 || println(io, "  min group length:  ", spec.min_group_length)
    spec.factorisation === :any || println(io, "  factorisation:     ", spec.factorisation)
    isempty(spec.initial_messages) || println(io, "  initial messages:  ", join(map(p -> "$(first(p)) => $(last(p))", spec.initial_messages), ", "))
    print(io, "  defined:           ", spec.file, ":", spec.line)
    return nothing
end

Base.show(io::IO, ::MIME"text/html", spec::NodeSpec) = html_table(
    io, "NodeSpec: $(spec.node) ($(type_label(spec)))", [
        "interfaces" => join(map(interface_label, spec.interfaces), ", "),
        "default algorithm" => spec.algorithm, "static inputs" => spec.static_inputs,
        "defined" => "$(spec.file):$(spec.line)",
    ]
)

dependency_target_label(entry::TargetDependencies) = entry.indexed ? "(:$(entry.edge), k)" : ":$(entry.edge)"
function dependency_inputs_label(entry::TargetDependencies)
    labels = [entry.default ? ["default"] : String[]; map(dependency_label, collect(entry.inputs))]
    return isempty(labels) ? "nothing" : join(labels, ", ")
end
partition_label(spec::DependenciesSpec) =
    spec.partition === nothing ? "from the factorisation" : join(map(repr, spec.partition), ", ")

Base.show(io::IO, spec::DependenciesSpec) = print(io, "DependenciesSpec(", spec.node, ", ", spec.algorithm, ")")

function Base.show(io::IO, ::MIME"text/plain", spec::DependenciesSpec)
    println(io, "DependenciesSpec: ", spec.node, " under ", spec.algorithm)
    width = maximum(length ∘ dependency_target_label, spec.targets; init = 0)
    for entry in spec.targets
        println(io, "  ", rpad(dependency_target_label(entry), width), " ⇐ ", dependency_inputs_label(entry))
    end
    print(io, "  free-energy partition: ", partition_label(spec))
    return nothing
end

Base.show(io::IO, ::MIME"text/html", spec::DependenciesSpec) = html_table(
    io, "DependenciesSpec: $(spec.node) under $(spec.algorithm)",
    [
        [dependency_target_label(entry) => dependency_inputs_label(entry) for entry in spec.targets];
        ["free-energy partition" => partition_label(spec)]
    ]
)

Base.show(io::IO, coverage::RuleCoverage) = print(io, "RuleCoverage(", coverage.node, ")")

coverage_cell(coverage, row, algorithm) =
    (n = get(coverage.counts, (row, algorithm), 0); n == 0 ? "" : n == 1 ? "✓" : "✓×$n")

function Base.show(io::IO, ::MIME"text/plain", coverage::RuleCoverage)
    println(io, "Rule coverage for ", coverage.node)
    names = map(string ∘ nameof, coverage.algorithms)
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
    print(io, "<table><caption>Rule coverage for ", html_escape(coverage.node), "</caption><tr><th></th>")
    foreach(a -> print(io, "<th>", html_escape(nameof(a)), "</th>"), coverage.algorithms)
    print(io, "</tr>")
    for row in coverage.rows
        print(io, "<tr><th style=\"text-align:left\">", html_escape(row), "</th>")
        foreach(a -> print(io, "<td>", coverage_cell(coverage, row, a), "</td>"), coverage.algorithms)
        print(io, "</tr>")
    end
    print(io, "</table>")
    return nothing
end
