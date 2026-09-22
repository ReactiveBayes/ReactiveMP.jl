# Every module that defines rules or nodes owns one registry, a `const` created by the
# definition macros and filled at the module's top level. A module's precompile image then
# carries its own entries; nothing is ever pushed into a registry owned by this package,
# which would land only in the image of whichever package happened to precompile it.

const REGISTRY_NAME = :__message_passing_registry__

"""
    Registry

The rules and nodes one module defines.
"""
struct Registry
    rules::Vector{RuleSpec}
    nodes::Vector{Any}
end

Registry() = Registry(RuleSpec[], Any[])

"""
    @define_registry

Create the calling module's registry unless it exists. Emitted by the definition macros.
"""
macro define_registry()
    return esc(:(isdefined(@__MODULE__, $(QuoteNode(REGISTRY_NAME))) || (const $REGISTRY_NAME = $Registry())))
end

rulekey(spec::RuleSpec) = (spec.kind, spec.node, spec.target, spec.algorithm, spec.signature)

"""
    register!(registry, spec::RuleSpec)

Add a rule, replacing one with the same signature, so redefining a rule at the REPL
updates it rather than duplicating it.
"""
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

"""
    registries()

Every registry in the loaded modules and their submodules, including `Main`.
"""
function registries()
    found = Pair{Module, Registry}[]
    visited = Set{Module}()
    roots = Module[Main; collect(values(Base.loaded_modules))]
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
    registered_rules()

Every rule defined in the loaded modules.
"""
registered_rules() = RuleSpec[spec for (_, registry) in registries() for spec in registry.rules]

"""
    duplicate_rules()

Rules with identical signatures defined in more than one module, grouped by signature.
Within one module Julia itself rejects the second definition during precompilation;
across modules the later method silently replaces the earlier, which this reports.
Ambiguity — overlapping but different signatures — is a separate question, answered by
`check_rule_ambiguities`.
"""
function duplicate_rules()
    groups = Dict{Any, Vector{Pair{Module, RuleSpec}}}()
    for (mod, registry) in registries(), spec in registry.rules
        push!(get!(groups, rulekey(spec), Pair{Module, RuleSpec}[]), mod => spec)
    end
    return [entries for entries in values(groups) if length(entries) > 1]
end
