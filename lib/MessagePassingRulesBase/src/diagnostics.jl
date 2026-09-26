# What is wrong when no rule fits, and what is inconsistent among the rules that exist.

function input_label(container, key, selection = :single)
    selection === :cluster && length(key) == 1 && return "$container[($(repr(only(key))),)]"
    selection === :cluster && return "$container[$(join(repr.(key), ", "))]"
    selection === :all && return "$container[:$key...]"
    selection === :aligned && return "$container[:$key][k]"
    selection === :allbutself && return "$container[:$key][!k]"
    return "$container[:$key]"
end

function provided_inputs(args::RuleArgs)
    provided = Tuple{Symbol, Any, Type}[]
    for (key, value) in pairs(args.m.values)
        push!(provided, (:m, key, typeof(value)))
    end
    for (key, value) in pairs(args.q.singles)
        push!(provided, (:q, key, typeof(value)))
    end
    for (key, value) in zip(joint_keys(args.q), args.q.joints)
        push!(provided, (:q, key, typeof(value)))
    end
    return provided
end
provided_inputs(args) = nothing

joint_keys(::Marginals{N, T, J}) where {N, T, J} = J

function input_accepts(input::InputSpec, type::Type)
    input.selection === :all && return type <: Tuple{Vararg{input.type}}
    input.selection in (:aligned, :allbutself) && return type <: Tuple{Vararg{Union{Nothing, input.type}}}
    return type <: input.type
end

node_matches(spec, node) = node isa node_dispatch_type(spec.node)

function candidate_rules(notfound::RuleNotFound)
    return filter(registered_rules()) do spec
        spec.kind === notfound.kind && node_matches(spec, notfound.node) &&
            (notfound.target === nothing ? spec.target === Nothing : notfound.target isa spec.target)
    end
end

function Base.showerror(io::IO, err::RuleNotFoundError)
    nf = err.notfound
    print(io, "no ", nf.kind, " rule for ", nf.node)
    nf.target === nothing || print(io, " towards ", nf.target)
    print(io, " under ", nf.algorithm)
    provided = provided_inputs(nf.args)
    if provided === nothing
        print(io, " for arguments of type ", typeof(nf.args))
        return nothing
    end
    labels = [input_label(c, k, k isa Tuple ? :cluster : :single) * "::" * string(t) for (c, k, t) in provided]
    print(io, " takes the inputs (", join(labels, ", "), ")")

    candidates = candidate_rules(nf)
    if isempty(candidates)
        print(io, "\n  no rule exists for this node and target under any algorithm")
        return nothing
    end
    provided_keys = Set((c, k) for (c, k, _) in provided)
    same_shape = filter(spec -> Set((i.container, i.key) for i in spec.inputs) == provided_keys, candidates)
    if any(spec -> admits(nf.algorithm, spec.algorithm), same_shape)
        print(io, "\n  a rule of this shape exists, but the input types do not fit (type mismatch)")
    else
        print(io, "\n  no rule consumes this set of inputs under this algorithm (no rule of this shape)")
    end
    print(io, "\n  near misses:")
    for spec in candidates
        print(io, "\n    rule at ", spec.file, ":", spec.line)
        for (ok, text) in fit_report(spec, nf.algorithm, provided)
            print(io, "\n      ", ok ? "✓" : "✗", " ", text)
        end
    end
    return nothing
end

# How a rule fits a call, slot by slot: its algorithm, each of its inputs against what was
# provided, and each provided input it does not consume, as `(fits, description)` pairs.
function fit_report(spec::RuleSpec, algorithm, provided)
    lines = Tuple{Bool, String}[(admits(algorithm, spec.algorithm), "algorithm $(spec.algorithm)")]
    for input in spec.inputs
        found = findfirst(((c, k, _),) -> c === input.container && k == input.key, provided)
        label = input_label(input.container, input.key, input.selection) * "::" * string(input.type)
        if found === nothing
            push!(lines, (false, "$label  not provided"))
        else
            type = provided[found][3]
            push!(lines, (input_accepts(input, type), "$label  got $type"))
        end
    end
    for (c, k, t) in provided
        any(i -> i.container === c && i.key == k, spec.inputs) && continue
        push!(lines, (false, "$(input_label(c, k, k isa Tuple ? :cluster : :single))::$t  provided but not consumed"))
    end
    return lines
end

"""
    RuleIssue

A problem [`check_rules`](@ref) found with one rule. Fields: `rule`, the [`RuleSpec`](@ref), and
`message`, a sentence saying what is wrong. It shows itself as
`RuleIssue(file:line: message)`.
"""
struct RuleIssue
    rule::RuleSpec
    message::String
end

Base.show(io::IO, issue::RuleIssue) =
    print(io, "RuleIssue(", issue.rule.file, ":", issue.rule.line, ": ", issue.message, ")")

"""
    check_rules(modules::Module...) -> Vector{RuleIssue}

Check every rule defined in `modules`, by default in every loaded module, against its node's
declaration and the dependency declarations, and return the problems as [`RuleIssue`](@ref)s;
an empty vector means none. It checks:

- that the rule's node is declared;
- its target: an interface of the node, a group written `(:m, k)` and a single interface not,
  and a cluster's members existing, in interface order, a group's members by index, and not a
  lone single interface, whose marginal is written `q[:x]`;
- each input: an existing interface, a group selected as `[:m...]`, `[k]` or `[!k]` and a single
  interface not, and a cluster as for the target;
- for a message rule whose algorithm has a dependency declaration (its own, or for a
  [`DefaultAlgorithmExtension`](@ref) that declares none, the default's) listing its target:
  that it consumes exactly the declared inputs, or, for a target declared with `default`, at
  least the added ones. A rule declared with `default` in its `args` is not compared.

A package's tests call it on the package's own module.
"""
function check_rules(modules::Module...)
    issues = RuleIssue[]
    declarations = registered_dependencies()
    for spec in registered_rules(modules...)
        applicable(nodespec, spec.node) ||
            (push!(issues, RuleIssue(spec, "its node $(spec.node) has no declaration")); continue)
        node = nodespec(spec.node)
        for message in rule_problems(spec, node, declarations)
            push!(issues, RuleIssue(spec, message))
        end
    end
    return issues
end

function rule_problems(spec::RuleSpec, node::NodeSpec, declarations)
    problems = String[]
    names = map(i -> i.name, node.interfaces)
    isgroup(name) = any(i -> i.name === name && i.group, node.interfaces)
    # A cluster's members are interfaces, or members of a group, `(:T, 1)`, listed in interface
    # order and a group's members by index.
    member_name(m) = m isa Symbol ? m : first(m)
    function cluster_problem(members)
        for m in members
            member_name(m) in names || return "a cluster lists existing interfaces in interface order, $(Tuple(names))"
            m isa Tuple && !isgroup(member_name(m)) && return "`$(member_name(m))` is not a group, so it has no member $(last(m))"
        end
        positions = map(m -> (findfirst(==(member_name(m)), names), m isa Tuple ? last(m) : 0), members)
        (issorted(positions) && allunique(positions)) || return "a cluster lists existing interfaces in interface order, $(Tuple(names)), and a group's members by index"
        return nothing
    end
    lone_single(members) = length(members) == 1 && only(members) in names && !isgroup(only(members))
    lone_message(members) = "a one-member cluster of a single interface is its marginal; write `q[$(repr(only(members)))]`"

    target = spec.target
    if target <: Target
        edge = target.parameters[1]
        if !(edge in names)
            push!(problems, "`target = :$edge`: $(spec.node) has no interface `$edge`")
        elseif isgroup(edge)
            push!(problems, "`target = :$edge`: `$edge` is a group; its targets are written `(:$edge, k)`")
        end
    elseif target <: IndexedTarget
        edge = target.parameters[1]
        (edge in names && isgroup(edge)) ||
            push!(problems, "`target = (:$edge, k)`: $(spec.node) has no group `$edge`")
    elseif target === ClusterTarget
        # A marginal rule over any cluster: no members to check.
    elseif target <: ClusterTarget
        members = target.parameters[1]
        if lone_single(members)
            push!(problems, "`target = $members`: $(lone_message(members))")
        else
            problem = cluster_problem(members)
            problem === nothing || push!(problems, "`target = $members`: $problem")
        end
    end

    for input in spec.inputs
        label = input_label(input.container, input.key, input.selection)
        if input.selection === :cluster && lone_single(input.key)
            push!(problems, "`$label`: $(lone_message(input.key))")
        elseif input.selection === :cluster
            problem = cluster_problem(input.key)
            problem === nothing || push!(problems, "`$label`: $problem")
        elseif !(input.key in names)
            push!(problems, "`$label`: $(spec.node) has no interface `$(input.key)`")
        elseif input.selection === :single && isgroup(input.key)
            push!(problems, "`$label`: `$(input.key)` is a group; consume it as `[:$(input.key)...]`, `[k]` or `[!k]`")
        elseif input.selection !== :single && !isgroup(input.key)
            push!(problems, "`$label`: `$(input.key)` is not a group")
        end
    end

    # The declaration a rule answers to: its own algorithm's, or for an extension that declares
    # none, the default's.
    own = filter(d -> d.node === spec.node && spec.algorithm <: d.algorithm, declarations)
    if isempty(own) && spec.algorithm <: DefaultAlgorithmExtension
        own = filter(d -> d.node === spec.node && d.algorithm === DefaultAlgorithm, declarations)
    end
    for declaration in own
        spec.kind === :message || continue
        # A rule over the default scheme's inputs consumes whatever the factorisation delivers.
        spec.default && continue
        instance = target <: IndexedTarget ? target(1) : target()
        entry = target_entry(declaration, instance)
        entry === nothing && continue
        expected = Set(dependency_label(d) for d in entry.inputs)
        actual = Set(input_label(i.container, i.key, i.selection) for i in spec.inputs)
        consumed = "consumes ($(join(sort(collect(actual)), ", "))) but the dependencies of $(nameof(declaration.algorithm))"
        # Beside `default`, the inputs follow the factorisation, which a rule does not know: only
        # the added ones are checked.
        if entry.default
            issubset(expected, actual) ||
                push!(problems, "$consumed add ($(join(sort(collect(expected)), ", "))) to the default scheme's inputs")
        else
            expected == actual || push!(problems, "$consumed declare ($(join(sort(collect(expected)), ", ")))")
        end
    end
    return problems
end

function dependency_label(d::Dependency)
    d.key isa Tuple && return input_label(d.container, d.key, :cluster)
    d.selector isa SingleInterface && return input_label(d.container, d.key, :single)
    d.selector isa AllGroupMembers && return input_label(d.container, d.key, :all)
    d.selector isa AlignedGroupMember && return input_label(d.container, d.key, :aligned)
    d.selector isa AllGroupMembersButSelf && return input_label(d.container, d.key, :allbutself)
    return input_label(d.container, d.key, :all)
end

function full_signature(spec::RuleSpec)
    node = node_dispatch_type(spec.node)
    spec.kind === :average_energy && return Tuple{node, spec.algorithm, spec.signature}
    return Tuple{node, spec.target, spec.algorithm, spec.signature}
end

"""
    check_rule_ambiguities(modules::Module...) -> Vector{Tuple{RuleSpec, RuleSpec}}

The pairs of rules defined in `modules`, by default in every loaded module, that some call could
match equally well, so that resolution would throw a `MethodError`; an empty vector means none.
Only rules consuming the same set of inputs can overlap, so candidates are grouped by kind,
node, target and input names first. Within a group, each rule's own method of
[`find_message_rule`](@ref), [`find_marginal_rule`](@ref) or [`find_average_energy`](@ref) is
compared with Julia's `Base.isambiguous`.

Julia's check, not a hand-made `typeintersect`, because the signatures bound their inputs as
`Messages{N, <:Tuple{…}}`: when a slot is disjoint, the intersection is a valid but empty
type such as `Messages{N, Union{}}` or `PointMass{Union{}}`, never `Union{}` itself, so an
intersection test reports every disjoint pair. `Base.isambiguous` ignores ambiguities only a
`Union{}` parameter could trigger.
"""
function check_rule_ambiguities(modules::Module...)
    groups = Dict{Any, Vector{RuleSpec}}()
    for spec in registered_rules(modules...)
        key = (spec.kind, spec.node, spec.target, spec.default, Set((i.container, i.key) for i in spec.inputs))
        push!(get!(groups, key, RuleSpec[]), spec)
    end
    ambiguous = Tuple{RuleSpec, RuleSpec}[]
    for specs in values(groups), i in eachindex(specs), j in (i + 1):lastindex(specs)
        Base.isambiguous(rule_method(specs[i]), rule_method(specs[j])) && push!(ambiguous, (specs[i], specs[j]))
    end
    return ambiguous
end

rule_function(spec::RuleSpec) = spec.kind === :message ? find_message_rule : spec.kind === :marginal ? find_marginal_rule : find_average_energy

# The method the definition macro added for `spec`: the one whose signature is the spec's.
# Not finding it is a bug, never a reason to skip the pair and hide an ambiguity.
function rule_method(spec::RuleSpec)
    f = rule_function(spec)
    signature = Tuple{typeof(f), full_signature(spec).parameters...}
    for method in methods(f)
        method.sig == signature && return method
    end
    return error("no method of $f has the signature of $spec")
end
