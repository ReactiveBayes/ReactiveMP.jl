
using TupleTools

"""
    MetaSpecificationEntry{F, N, M}

This structure is used with a combination of `@meta` macro from GraphPPL.jl. 

# Arguments
- `F`: functional form of a node in a form of the `Symbol`
- `N`: names of variables for the meta specification
- `meta::M`: meta object
"""
struct MetaSpecificationEntry{F, N, M}
    meta :: M

    MetaSpecificationEntry(::Val{F}, ::Val{N}, meta::M) where { F, N, M } = new{F, N, M}(meta)
end

functionalform(entry::MetaSpecificationEntry{F}) where { F } = F
getnames(entry::MetaSpecificationEntry{F, N}) where { F, N } = N
metadata(entry::MetaSpecificationEntry)                      = entry.meta

function Base.show(io::IO, entry::MetaSpecificationEntry{F, N}) where { F, N }
    print(io, F, "(")
    join(io, N, ", ")
    print(io, ") = ", entry.meta)
end


struct MetaSpecification{E}
    entries :: E
end

getentries(specification::MetaSpecification) = specification.entries

function Base.show(io::IO, specification::MetaSpecification)
    print(io, "Meta specification:\n\t")
    join(io, specification.entries, "\n\t")
end

"""
    resolve_meta(expr::Expr, fform::Symbol, variables, metaspec, model) 

This function resolves meta for a given `expr` (needed for error printing), `fform`, `variables`, `constraints` and `model`.

See also: [`ConstraintsSpecification`](@ref)
"""
function resolve_meta(expr::Expr, fform::Symbol, variables, metaspec, model) 
    var_refs       = map(resolve_variable_proxy, variables)
    var_refs_names = map(r -> r[1], var_refs)

    found = nothing

    unrolled_foreach(getentries(metaspec)) do fentry
        if functionalform(fentry) === fform && all(s -> s ∈ var_refs_names, getnames(fentry))
            if found !== nothing
                error("Ambigous meta object resolution in the expression $(expr). Check $(found) and $(fentry).")
            end
            found = fentry
        end
    end

    return found === nothing ? nothing : metadata(found)

end