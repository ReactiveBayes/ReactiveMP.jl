export ConstraintsSpecification

import Base: show

const ConstraintsSpecificationPreallocatedDefaultSize = 64

struct ConstraintsSpecificationPreallocated
    clusters_template  :: BitVector
    clusters_usage     :: BitVector
    clusters_set       :: Set{Tuple}
    cluster_indices    :: Vector{Int}
    var_refs_positions :: Vector{Int}

    ConstraintsSpecificationPreallocated() = new(
        trues(ConstraintsSpecificationPreallocatedDefaultSize),
        falses(ConstraintsSpecificationPreallocatedDefaultSize),
        Set{Tuple}(),
        Vector{Int}(undef, ConstraintsSpecificationPreallocatedDefaultSize),
        Vector{Int}(undef, ConstraintsSpecificationPreallocatedDefaultSize)
    )
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
    warn::Bool
end

iswarn(options::ConstraintsSpecificationOptions) = options.warn

struct ConstraintsSpecification{F, M, S}
    factorisation :: F
    marginalsform :: M
    messagesform  :: S
    options       :: ConstraintsSpecificationOptions
    preallocated  :: ConstraintsSpecificationPreallocated
end

ConstraintsSpecification(
    factorisation::F,
    marginalsform::M,
    messagesform::S,
    options::ConstraintsSpecificationOptions
) where {F, M, S} = ConstraintsSpecification{F, M, S}(
    factorisation,
    marginalsform,
    messagesform,
    options,
    ConstraintsSpecificationPreallocated()
)

getoptions(specification::ConstraintsSpecification) = specification.options

struct UnspecifiedConstraints end

const DefaultConstraints = UnspecifiedConstraints()

const __EmptyConstraints = ConstraintsSpecification((), (;), (;), ConstraintsSpecificationOptions(true))

__reset_preallocated!(specification::ConstraintsSpecification, size::Int) =
    __reset_preallocated!(specification.preallocated, size)

function activate!(constraints::UnspecifiedConstraints, model)
    return nothing
end

function activate!(constraints::ConstraintsSpecification, model)
    options = getoptions(constraints)
    warn    = iswarn(options)

    # Check functional forms first 

    # Check marginal form constraints
    foreach(pairs(constraints.marginalsform)) do entry
        specname = first(entry)
        # Check if form constrain applied for datavar or constvar
        if warn && (hasdatavar(model, specname) || hasconstvar(model, specname))
            @warn "Constraints specification has marginal form constraint for `q($(specname))`, but `$(specname)` is not a random variable. It is not possible to set a form constrain on non-random variable. Form constraint is ignored. Use `warn = false` option during constraints specification to suppress this warning."
            # Check if variable does not exist 
        elseif warn && !hasrandomvar(model, specname)
            @warn "Constraints specification has marginal form constraint for `q($(specname))`, but model has no random variable named `$(specname)`. Use `warn = false` option during constraints specification to suppress this warning."
        end
    end

    # Check messages form constraints
    foreach(pairs(constraints.messagesform)) do entry
        specname = first(entry)
        # Check if form constrain applied for datavar or constvar
        if warn && (hasdatavar(model, specname) || hasconstvar(model, specname))
            @warn "Constraints specification has messages form constraint for `μ($(specname))`, but `$(specname)` is not a random variable. It is not possible to set a form constrain on non-random variable. Form constraint is ignored. Use `warn = false` option during constraints specification to suppress this warning."
            # Check if variable does not exist 
        elseif warn && !hasrandomvar(model, specname)
            @warn "Constraints specification has messages form constraint for `μ($(specname))`, but model has no random variable named `$(specname)`. Use `warn = false` option during constraints specification to suppress this warning."
        end
    end

    # Check LHS of factorisation constraints
    foreach(constraints.factorisation) do spec
        specnames = getnames(spec)
        foreach(specnames) do specname
            if warn && (hasdatavar(model, specname) || hasconstvar(model, specname))
                @warn "Constraints specification has factorisation constraint for `q($(join(specnames, ", ")))`, but `$(specname)` is not a random variable. Data variables and constants in the model are forced to be factorized by default such that `q($(join(specnames, ", "))) = q($(specname))q(...)` . Use `warn = false` option during constraints specification to suppress this warning."
            elseif warn && !hasrandomvar(model, specname)
                @warn "Constraints specification has factorisation constraint for `q($(join(specnames, ", ")))`, but model has no random variable named `$(specname)`. Use `warn = false` option during constraints specification to suppress this warning."
            end
        end
    end

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
        print(io, "    μ(", first(spec), ") :: ", last(spec), "\n")
    end
    print(io, "  factorisation:\n")
    foreach(specification.factorisation) do f
        print(io, "    ", f, "\n")
    end
    print(io, "Options:\n")
    print(io, "  warn = ", specification.options.warn, "\n")
end
