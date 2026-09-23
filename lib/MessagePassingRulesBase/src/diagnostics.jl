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
    if any(spec -> nf.algorithm isa spec.algorithm, same_shape)
        print(io, "\n  a rule of this shape exists, but the input types do not fit (type mismatch)")
    else
        print(io, "\n  no rule consumes this set of inputs under this algorithm (no rule of this shape)")
    end
    print(io, "\n  near misses:")
    for spec in candidates
        print(io, "\n    rule at ", spec.file, ":", spec.line)
        mark(ok) = ok ? "✓" : "✗"
        print(io, "\n      ", mark(nf.algorithm isa spec.algorithm), " algorithm ", spec.algorithm)
        for input in spec.inputs
            found = findfirst(((c, k, _),) -> c === input.container && k == input.key, provided)
            label = input_label(input.container, input.key, input.selection) * "::" * string(input.type)
            if found === nothing
                print(io, "\n      ✗ ", label, "  not provided")
            else
                type = provided[found][3]
                print(io, "\n      ", mark(input_accepts(input, type)), " ", label, "  got ", type)
            end
        end
        for (c, k, t) in provided
            any(i -> i.container === c && i.key == k, spec.inputs) && continue
            print(io, "\n      ✗ ", input_label(c, k, k isa Tuple ? :cluster : :single), "::", t, "  provided but not consumed")
        end
    end
    return nothing
end

"""
    RuleIssue

A problem [`check_rules`](@ref) found with one rule.
"""
struct RuleIssue
    rule::RuleSpec
    message::String
end

Base.show(io::IO, issue::RuleIssue) =
    print(io, "RuleIssue(", issue.rule.file, ":", issue.rule.line, ": ", issue.message, ")")

"""
    check_rules([modules...])

Check every rule in `modules` (by default all loaded) against its node's declaration and
its algorithm's dependency declaration. Returns the problems as [`RuleIssue`](@ref)s; an
empty vector means none.
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
    order_ok(members) = all(in(names), members) && issorted(map(m -> findfirst(==(m), names), members))
    lone_single(members) = length(members) == 1 && only(members) in names && !isgroup(only(members))
    lone_message(members) = "a one-member cluster of a single interface is its marginal; write `q[$(repr(only(members)))]`"

    target = spec.target
    if target <: Target
        edge = target.parameters[1]
        if !(edge in names)
            push!(problems, "`towards = :$edge`: $(spec.node) has no interface `$edge`")
        elseif isgroup(edge)
            push!(problems, "`towards = :$edge`: `$edge` is a group; its targets are written `(:$edge, k)`")
        end
    elseif target <: IndexedTarget
        edge = target.parameters[1]
        (edge in names && isgroup(edge)) ||
            push!(problems, "`towards = (:$edge, k)`: $(spec.node) has no group `$edge`")
    elseif target <: ClusterTarget
        members = target.parameters[1]
        if lone_single(members)
            push!(problems, "`towards = $members`: $(lone_message(members))")
        elseif !order_ok(members)
            push!(problems, "`towards = $members`: a cluster lists existing interfaces in interface order, $(Tuple(names))")
        end
    end

    for input in spec.inputs
        label = input_label(input.container, input.key, input.selection)
        if input.selection === :cluster && lone_single(input.key)
            push!(problems, "`$label`: $(lone_message(input.key))")
        elseif input.selection === :cluster
            order_ok(input.key) ||
                push!(problems, "`$label`: a cluster lists existing interfaces in interface order, $(Tuple(names))")
        elseif !(input.key in names)
            push!(problems, "`$label`: $(spec.node) has no interface `$(input.key)`")
        elseif input.selection === :single && isgroup(input.key)
            push!(problems, "`$label`: `$(input.key)` is a group; consume it as `[:$(input.key)...]`, `[k]` or `[!k]`")
        elseif input.selection !== :single && !isgroup(input.key)
            push!(problems, "`$label`: `$(input.key)` is not a group")
        end
    end

    for declaration in declarations
        (declaration.node === spec.node && spec.algorithm <: declaration.algorithm) || continue
        spec.kind === :message || continue
        instance = target <: IndexedTarget ? target(1) : target()
        declared = target_dependencies(declaration, instance)
        declared === nothing && continue
        expected = Set(dependency_label(d) for d in declared)
        actual = Set(input_label(i.container, i.key, i.selection) for i in spec.inputs)
        expected == actual ||
            push!(problems, "consumes ($(join(sort(collect(actual)), ", "))) but the dependencies of $(declaration.algorithm) declare ($(join(sort(collect(expected)), ", ")))")
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
    check_rule_ambiguities([modules...])

Pairs of rules in `modules` (by default all loaded) that some call could match equally
well. Only rules consuming the same set of inputs can overlap, so candidates are grouped by
kind, node, target and input names first, and compared with `typeintersect` within a group.
"""
function check_rule_ambiguities(modules::Module...)
    groups = Dict{Any, Vector{RuleSpec}}()
    for spec in registered_rules(modules...)
        key = (spec.kind, spec.node, spec.target, Set((i.container, i.key) for i in spec.inputs))
        push!(get!(groups, key, RuleSpec[]), spec)
    end
    ambiguous = Tuple{RuleSpec, RuleSpec}[]
    for specs in values(groups), i in eachindex(specs), j in (i + 1):lastindex(specs)
        a, b = full_signature(specs[i]), full_signature(specs[j])
        typeintersect(a, b) === Union{} && continue
        (a <: b || b <: a) && continue
        push!(ambiguous, (specs[i], specs[j]))
    end
    return ambiguous
end
