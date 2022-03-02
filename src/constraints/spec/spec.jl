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

struct ConstraintsSpecification{F, M, S}
    factorisation :: F
    marginalsform :: M
    messagesform  :: S
    preallocated :: ConstraintsSpecificationPreallocated
end

ConstraintsSpecification(factorisation::F, marginalsform::M, messagesform::S) where { F, M, S } = ConstraintsSpecification{F, M, S}(factorisation, marginalsform, messagesform, ConstraintsSpecificationPreallocated())

const DefaultConstraints = ConstraintsSpecification((), (;), (;))

__reset_preallocated!(specification::ConstraintsSpecification, size::Int) = __reset_preallocated!(specification.preallocated, size)

function Base.show(io::IO, specification::ConstraintsSpecification) 
    print(io, "Constraints:\n")
    print(io, "\tmarginals form:\n")
    foreach(pairs(specification.marginalsform)) do spec 
        print(io, "\t\tq($(first(spec))) :: ")
        join(io, map(repr, last(spec)), " :: ")
        print(io, "\n")
    end
    print(io, "\tmessages form:\n")
    foreach(pairs(specification.messagesform)) do spec 
        print(io, "\t\tμ($(first(spec))) :: ")
        join(io, map(repr, last(spec)), " :: ")
        print(io, "\n")
    end
    print(io, "\tfactorisation:\n")
    foreach(specification.factorisation) do f
        print(io, "\t\t", f, "\n")
    end
end