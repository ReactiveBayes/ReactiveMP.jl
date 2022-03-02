

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

function Base.show(io::IO, entry::MetaSpecificationEntry{F, N}) where { F, N }
    print(io, F, "(")
    join(io, N, ", ")
    print(io, ") = ", entry.meta)
end


struct MetaSpecification{E}
    entries :: E
end

function Base.show(io::IO, specification::MetaSpecification)
    print(io, "Meta specification:\n\t")
    join(io, specification.entries, "\n\t")
end