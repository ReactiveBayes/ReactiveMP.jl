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

struct ConstraintsSpecification{F, M}
    factorisation :: F
    form :: M
    preallocated :: ConstraintsSpecificationPreallocated
end

ConstraintsSpecification(factorisation::F, form::M) where { F, M } = ConstraintsSpecification{F, M}(factorisation, form, ConstraintsSpecificationPreallocated())

__reset_preallocated!(specification::ConstraintsSpecification, size::Int) = __reset_preallocated!(specification.preallocated, size)

function Base.show(io::IO, specification::ConstraintsSpecification) 
    print(io, "Constraints:\n\tform: $(specification.form)\n")
    print(io, "\tfactorisation\n")
    foreach(specification.factorisation) do f
        print(io, "\t\t", f, "\n")
    end
end