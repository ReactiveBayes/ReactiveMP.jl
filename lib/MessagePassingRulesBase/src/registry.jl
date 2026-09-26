# Every module that defines rules or nodes owns one registry, a `const` created by the
# definition macros and filled at the module's top level. A module's precompile image then
# carries its own entries; nothing is ever pushed into a registry owned by this package,
# which would land only in the image of whichever package happened to precompile it.

const REGISTRY_NAME = :__message_passing_registry__

"""
    Registry

The rules, nodes and dependency declarations one module defines, in its fields `rules`
([`RuleSpec`](@ref)s), `nodes` ([`NodeSpec`](@ref)s) and `dependencies`
([`DependenciesSpec`](@ref)s). The definition macros create it in the defining module, as the
constant `__message_passing_registry__`, and fill it at the module's top level, so a package's
precompile image carries its own entries. A redefinition, at the REPL say, replaces the entry
with the same key.

It is for introspection only: listing, checking and displaying rules, and suggesting candidates
when no rule fits. Resolution never reads it; it is Julia's dispatch.

See also [`registries`](@ref), [`registered_rules`](@ref).
"""
struct Registry
    rules::Vector{RuleSpec}
    nodes::Vector{NodeSpec}
    dependencies::Vector{DependenciesSpec}
end

Registry() = Registry(RuleSpec[], NodeSpec[], DependenciesSpec[])

# Create the calling module's registry unless it exists. Emitted by the definition macros.
macro define_registry()
    return esc(:(isdefined(@__MODULE__, $(QuoteNode(REGISTRY_NAME))) || (const $REGISTRY_NAME = $Registry())))
end

rulekey(spec::RuleSpec) = (spec.kind, spec.node, spec.target, spec.algorithm, spec.signature)

# Add a rule, replacing one with the same signature, so redefining a rule at the REPL
# updates it rather than duplicating it.
function register!(registry::Registry, spec::RuleSpec)
    key = rulekey(spec)
    position = findfirst(existing -> rulekey(existing) == key, registry.rules)
    if position === nothing
        push!(registry.rules, spec)
    else
        registry.rules[position] = spec
    end
    return spec
end

# Add a node, replacing an earlier declaration of the same node.
function register!(registry::Registry, spec::NodeSpec)
    position = findfirst(existing -> existing.node === spec.node, registry.nodes)
    if position === nothing
        push!(registry.nodes, spec)
    else
        registry.nodes[position] = spec
    end
    return spec
end

# Add a dependency declaration, replacing an earlier one for the same node and algorithm.
function register!(registry::Registry, declaration::DependenciesSpec)
    position = findfirst(
        existing -> existing.node === declaration.node && existing.algorithm === declaration.algorithm,
        registry.dependencies,
    )
    position === nothing ? push!(registry.dependencies, declaration) : (registry.dependencies[position] = declaration)
    return declaration
end

"""
    registries(modules::Module...) -> Vector{Pair{Module, Registry}}

Every [`Registry`](@ref) in `modules` and their submodules, each with the module that owns it;
with no argument, in `Main` and every loaded module.
"""
function registries(modules::Module...)
    found = Pair{Module, Registry}[]
    visited = Set{Module}()
    roots = isempty(modules) ? Module[Main; collect(values(Base.loaded_modules))] : collect(modules)
    for root in roots
        collect_registries!(found, visited, root)
    end
    return found
end

function collect_registries!(found, visited, mod::Module)
    mod in visited && return found
    push!(visited, mod)
    if isdefined(mod, REGISTRY_NAME)
        registry = getfield(mod, REGISTRY_NAME)
        registry isa Registry && push!(found, mod => registry)
    end
    for name in names(mod; all = true, imported = false)
        isdefined(mod, name) || continue
        sub = getfield(mod, name)
        if sub isa Module && sub !== mod && parentmodule(sub) === mod
            collect_registries!(found, visited, sub)
        end
    end
    return found
end

"""
    registered_rules(modules::Module...) -> Vector{RuleSpec}

Every rule defined in `modules` and their submodules; with no argument, in every loaded module.
"""
registered_rules(modules::Module...) = RuleSpec[spec for (_, registry) in registries(modules...) for spec in registry.rules]

"""
    registered_nodes(modules::Module...) -> Vector{NodeSpec}

Every node declared in `modules` and their submodules; with no argument, in every loaded module.
"""
registered_nodes(modules::Module...) = NodeSpec[spec for (_, registry) in registries(modules...) for spec in registry.nodes]

"""
    registered_dependencies(modules::Module...) -> Vector{DependenciesSpec}

Every [`DependenciesSpec`](@ref) declared in `modules` and their submodules; with no argument, in
every loaded module.
"""
registered_dependencies(modules::Module...) =
    DependenciesSpec[spec for (_, registry) in registries(modules...) for spec in registry.dependencies]

"""
    duplicate_rules() -> Vector{Vector{Pair{Module, RuleSpec}}}

Rules with identical signatures defined in more than one loaded module, one group per
signature, each rule with its module. Within one module Julia itself rejects the second
definition during precompilation; across modules the later method silently replaces the
earlier, which this reports. Ambiguity, overlapping but different signatures, is a separate
question, answered by [`check_rule_ambiguities`](@ref).
"""
function duplicate_rules()
    groups = Dict{Any, Vector{Pair{Module, RuleSpec}}}()
    for (mod, registry) in registries(), spec in registry.rules
        push!(get!(groups, rulekey(spec), Pair{Module, RuleSpec}[]), mod => spec)
    end
    return [entries for entries in values(groups) if length(entries) > 1]
end
