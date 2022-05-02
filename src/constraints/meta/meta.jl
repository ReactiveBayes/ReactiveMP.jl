
using TupleTools

"""
    MetaSpecificationEntry{F, N, M}

This structure is used with a combination of `@meta` macro from GraphPPL.jl. 

# Arguments
- `F`: functional form of a node in a form of the `Symbol`
- `N`: names of variables for the meta specification
- `meta::M`: meta object
"""
struct MetaSpecificationEntry{N, M}
    fform :: Symbol
    names :: NTuple{N, Symbol}
    meta  :: M
end

MetaSpecificationEntry(::Val{F}, ::Val{N}, meta::M) where {F, N, M} = MetaSpecificationEntry(F, N, meta)

functionalform(entry::MetaSpecificationEntry) = entry.fform
getnames(entry::MetaSpecificationEntry)       = entry.names
metadata(entry::MetaSpecificationEntry)       = entry.meta

function Base.show(io::IO, entry::MetaSpecificationEntry)
    print(io, functionalform(entry), "(")
    join(io, getnames(entry), ", ")
    print(io, ") -> ", metadata(entry))
end

struct MetaSpecificationOptions
    warn::Bool
end

iswarn(options::MetaSpecificationOptions) = options.warn

struct MetaSpecification{E}
    entries::E
    options::MetaSpecificationOptions
end

getentries(specification::MetaSpecification) = specification.entries
getoptions(specification::MetaSpecification) = specification.options

function Base.show(io::IO, specification::MetaSpecification)
    print(io, "Meta specification:\n  ")
    join(io, specification.entries, "\n  ")
    print(io, "\nOptions:\n")
    print(io, "  warn = ", specification.options.warn)
end

struct UnspecifiedMeta end

const DefaultMeta = UnspecifiedMeta()

"""
    resolve_meta(expr::Expr, fform::Symbol, variables, metaspec, model) 

This function resolves meta for a given `expr` (needed for error printing), `fform`, `variables`, `constraints` and `model`.

See also: [`ConstraintsSpecification`](@ref)
"""
function resolve_meta(metaspec, model, fform, variables)
    symfform = as_node_symbol(fform)

    var_names      = map(name, variables)
    var_refs       = map(resolve_variable_proxy, variables)
    var_refs_names = map(r -> r[1], var_refs)

    found = nothing

    unrolled_foreach(getentries(metaspec)) do fentry
        if functionalform(fentry) === symfform &&
           (all(s -> s ∈ var_names, getnames(fentry)) || all(s -> s ∈ var_refs_names, getnames(fentry)))
            if found !== nothing
                error("Ambigous meta object resolution for the node $(fform). Check $(found) and $(fentry).")
            end
            found = fentry
        end
    end

    return found === nothing ? nothing : metadata(found)
end

resolve_meta(metaspec::UnspecifiedMeta, model, fform, variables) = nothing

## 

function activate!(meta::UnspecifiedMeta, model)
    return nothing
end

function activate!(meta::MetaSpecification, model)
    options = getoptions(meta)
    stats   = getstats(model)

    warn = iswarn(options)

    foreach(getentries(meta)) do entry
        if warn && !hasnodeid(stats, functionalform(entry))
            @warn "Meta specification `$(entry)` specifies node entry as `$(functionalform(entry))`, but model has no factor node `$(functionalform(entry))`. Use `warn = false` option during constraints specification to suppress this warning."
        end

        foreach(getnames(entry)) do ename
            if warn && !(hasrandomvar(model, ename) || hasdatavar(model, ename) || hasconstvar(model, ename))
                @warn "Meta specification `$(entry)` uses `$(ename)`, but model has no variable named `$(ename)`. Use `warn = false` option during constraints specification to suppress this warning."
            end
        end
    end

    return nothing
end
