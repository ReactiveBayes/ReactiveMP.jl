export ConstraintsSpecification

import Base: show

const ConstraintsSpecificationPreallocatedDefaultSize = 64

struct ConstraintsSpecificationPreallocated
    clusters_template   :: BitVector
    clusters_usage      :: BitVector
    clusters_set        :: Set{Tuple}
    cluster_indices     :: Vector{Int}
    var_refs_positions  :: Vector{Int}
    
    ConstraintsSpecificationPreallocated() = new(trues(ConstraintsSpecificationPreallocatedDefaultSize), falses(ConstraintsSpecificationPreallocatedDefaultSize), Set{Tuple}(), Vector{Int}(undef, ConstraintsSpecificationPreallocatedDefaultSize), Vector{Int}(undef, ConstraintsSpecificationPreallocatedDefaultSize))
end

function __reset_preallocated!(preallocated::ConstraintsSpecificationPreallocated, size::Int)
    abs2size = abs2(size)
    if length(preallocated.clusters_template) < abs2size
        resize!(preallocated.clusters_template, abs2size)
        resize!(preallocated.clusters_usage, abs2size) # note: we dont need `size^2` for this, just `size` should be enough, but just to avoid extra checks
        resize!(preallocated.var_refs_positions, abs2size)
    end
    
    if length(preallocated.cluster_indices) < size
        resize!(preallocated.cluster_indices, size)
    end
    
    fill!(preallocated.clusters_template, true)
    fill!(preallocated.clusters_usage, false)
    
    empty!(preallocated.clusters_set)
end

struct ConstraintsSpecificationOptions
    warn :: Bool
end

struct ConstraintsSpecification{F, M, S}
    factorisation :: F
    marginalsform :: M
    messagesform  :: S
    options       :: ConstraintsSpecificationOptions
    preallocated  :: ConstraintsSpecificationPreallocated
end

ConstraintsSpecification(factorisation::F, marginalsform::M, messagesform::S, options::ConstraintsSpecificationOptions) where { F, M, S } = ConstraintsSpecification{F, M, S}(factorisation, marginalsform, messagesform, options, ConstraintsSpecificationPreallocated())

struct UnspecifiedConstraints end

const DefaultConstraints = UnspecifiedConstraints()

const __EmptyConstraints = ConstraintsSpecification((), (;), (;), ConstraintsSpecificationOptions(true))

__reset_preallocated!(specification::ConstraintsSpecification, size::Int) = __reset_preallocated!(specification.preallocated, size)

function activate!(constraints::UnspecifiedConstraints, model) 
    return nothing
end

function activate!(constraints::ConstraintsSpecification, model)
    return nothing
end

function Base.show(io::IO, specification::ConstraintsSpecification) 
    print(io, "Constraints:\n")
    print(io, "  marginals form:\n")
    foreach(pairs(specification.marginalsform)) do spec 
        print(io, "    q(", first(spec), ") :: ", last(spec), "\n")
    end
    print(io, "  messages form:\n")
    foreach(pairs(specification.messagesform)) do spec 
        print(io, "    q(", first(spec), ") :: ", last(spec), "\n")
    end
    print(io, "  factorisation:\n")
    foreach(specification.factorisation) do f
        print(io, "    ", f, "\n")
    end
    print(io, "  options:\n")
    print(io, "    warn = ", specification.options.warn, "\n")
end