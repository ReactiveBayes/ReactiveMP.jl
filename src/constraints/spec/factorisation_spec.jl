using Unrolled

"""
    CombinedRange{L, R}

`CombinedRange` represents a range of combined variable in factorisation specification language. Such variables specified to be in the same factorisation cluster.

See also: [`ReactiveMP.SplittedRange`](@ref)
"""
struct CombinedRange{L, R}
    from :: L
    to   :: R
end

Base.firstindex(range::CombinedRange) = range.from
Base.lastindex(range::CombinedRange)  = range.to
Base.in(item, range::CombinedRange)   = firstindex(range) <= item <= lastindex(range)

Base.show(io::IO, range::CombinedRange) = print(io, repr(range.from), ":", repr(range.to))

## 

"""
    SplittedRange{L, R}

`SplittedRange` represents a range of splitted variable in factorisation specification language. Such variables specified to be **not** in the same factorisation cluster.

See also: [`ReactiveMP.CombinedRange`](@ref)
"""
struct SplittedRange{L, R}
    from :: L
    to   :: R
end

is_splitted(any)                  = false
is_splitted(range::SplittedRange) = true

Base.firstindex(range::SplittedRange) = range.from
Base.lastindex(range::SplittedRange)  = range.to
Base.in(item, range::SplittedRange)   = firstindex(range) <= item <= lastindex(range)

Base.show(io::IO, range::SplittedRange) = print(io, repr(range.from), "..", repr(range.to))

## 

"""
    __as_unit_range

Converts a value to a `UnitRange`. This function is a part of private API and is not intended for public usage.
"""
function __as_unit_range end

__as_unit_range(any) = error("Internal error: Cannot represent $(any) as unit range.")
__as_unit_range(index::Integer)       = index:index
__as_unit_range(range::CombinedRange) = firstindex(range):lastindex(range)
__as_unit_range(range::SplittedRange) = firstindex(range):lastindex(range)

## 

"""
    FactorisationSpecificationNotDefinedYet{S}

Internal structure to denote not defined factorisation specification in constraints specification language. 
**Note**: this structure is a part of private API and is not intended for public use.
"""
struct FactorisationSpecificationNotDefinedYet{S} end

## 

struct FactorisationConstraintsEntry{N, I} end

FactorisationConstraintsEntry(::Val{N}, ::Val{I}) where { N, I } = FactorisationConstraintsEntry{N, I}()

getnames(entry::FactorisationConstraintsEntry{N})      where N        = N
getindices(entry::FactorisationConstraintsEntry{N, I}) where { N, I } = I
getpairs(entry::FactorisationConstraintsEntry)                        = zip(getnames(entry), getindices(entry))

__io_entry_pair(pair::Tuple)                            = __io_entry_pair(pair[1], pair[2])
__io_entry_pair(symbol::Symbol, ::Nothing)              = string(symbol)
__io_entry_pair(symbol::Symbol, index::Integer)         = string(symbol, "[", index, "]")
__io_entry_pair(symbol::Symbol, index::FunctionalIndex) = string(symbol, "[", repr(index), "]")
__io_entry_pair(symbol::Symbol, range::CombinedRange)   = string(symbol, "[", range, "]")
__io_entry_pair(symbol::Symbol, range::SplittedRange)   = string(symbol, "[", range, "]")

function Base.show(io::IO, entry::FactorisationConstraintsEntry)
    print(io, "q(")
    join(io, Iterators.map(__io_entry_pair, getpairs(entry)), ", ")
    print(io, ")")
end

## 

struct FactorisationConstraintsSpecification{N, E} end

FactorisationConstraintsSpecification(::Val{N}, ::Val{E})       where { N, E }       = FactorisationConstraintsSpecification{N, E}()
FactorisationConstraintsSpecification(::Val{N}, ::Val{nothing}) where { N          } = error("Cannot create q(", join(N, ","), ") factorisation constraints specification")

getnames(specification::FactorisationConstraintsSpecification{N})      where N        = N
getentries(specification::FactorisationConstraintsSpecification{N, E}) where { N, E } = E

Base.:(*)(left::Tuple{Vararg{T where T <: Union{ <:FactorisationConstraintsSpecification, <:FactorisationConstraintsEntry }}}, right::Tuple{Vararg{T where T <: Union{ <:FactorisationConstraintsSpecification, <:FactorisationConstraintsEntry }}}) = (left..., right...)
Base.:(*)(left::Union{ <:FactorisationConstraintsSpecification, <:FactorisationConstraintsEntry }, right::Tuple{Vararg{T where T <: Union{ <:FactorisationConstraintsSpecification, <:FactorisationConstraintsEntry }}}) = (left, right...)
Base.:(*)(left::Tuple{Vararg{T where T <: Union{ <:FactorisationConstraintsSpecification, <:FactorisationConstraintsEntry }}}, right::Union{ <:FactorisationConstraintsSpecification, <:FactorisationConstraintsEntry }) = (left..., right)
Base.:(*)(left::Union{ <:FactorisationConstraintsSpecification, <:FactorisationConstraintsEntry }, right::Union{ <:FactorisationConstraintsSpecification, <:FactorisationConstraintsEntry }) = (left, right)

Base.:(*)(::FactorisationSpecificationNotDefinedYet{S}, something::Any)                                 where S          = error("Cannot multiply $S and $something. $S has not been defined yet.")
Base.:(*)(something::Any, ::FactorisationSpecificationNotDefinedYet{S})                                 where S          = error("Cannot multiply $S and $something. $S has not been defined yet.")
Base.:(*)(::FactorisationSpecificationNotDefinedYet{S1}, ::FactorisationSpecificationNotDefinedYet{S2}) where { S1, S2 } = error("Cannot multiply $S1 and $S2. Both $S1 and $S2 have not been defined yet.")

function Base.show(io::IO, factorisation::FactorisationConstraintsSpecification{Names}) where Names
    
    print(io, "q(")
    join(io, getnames(factorisation), ", ")
    print(io, ")")
    
    iscompact = get(io, :compact, false)
    
    if !iscompact
        print(io, " = ")
        foreach(getentries(factorisation)) do e
            print(IOContext(io, :compact => true), e)
        end
    end

end

# Split related functions

Base.:(*)(left::Tuple{Vararg{T where T <: FactorisationConstraintsEntry}}, right::Tuple{Vararg{T where T <: FactorisationConstraintsEntry}}) = (left..., right...)

# Only these combinations are allowed to be merged
__factorisation_split_merge_range(a::Int, b::Int)                         = SplittedRange(a, b)
__factorisation_split_merge_range(a::FunctionalIndex, b::Int)             = SplittedRange(a, b)
__factorisation_split_merge_range(a::Int, b::FunctionalIndex)             = SplittedRange(a, b)
__factorisation_split_merge_range(a::FunctionalIndex, b::FunctionalIndex) = SplittedRange(a, b)
__factorisation_split_merge_range(a::Any, b::Any)                         = error("Cannot merge $(a) and $(b) indexes in `factorisation_split`")

function factorisation_split(left::Tuple{Vararg{T where T <: FactorisationConstraintsEntry}}, right::Tuple{Vararg{T where T <: FactorisationConstraintsEntry}})
    left_last   = last(left)
    right_first = first(right)
    (getnames(left_last) === getnames(right_first)) || error("Cannot split $(left_last) and $(right_first). Names or their order does not match.")
    (length(getnames(left_last)) === length(Set(getnames(left_last)))) || error("Cannot split $(left_last) and $(right_first). Names should be unique.")
    lindices = getindices(left_last)
    rindices = getindices(right_first)
    split_merged = unrolled_map(__factorisation_split_merge_range, lindices, rindices)
    
    # This check happens at runtime
    # first_split = first(split_merged)
    # unrolled_all(e -> e === first_split, split_merged) || error("Inconsistent indices within factorisation split. Check $(split_merged) indices for $(getnames(left_last)) variables.")
    
    return (left[1:end - 1]..., FactorisationConstraintsEntry(Val(getnames(left_last)), Val(split_merged)), right[begin+1:end]...)
end

## 

"""
    __factorisation_specification_resolve_index(index, collection)

This function materializes index from constraints specification to something we can use `Base.in` function to. For example constraint specification index may return `begin` or `end`
placeholders in a form of the `FunctionalIndex` structure. This function correctly resolves all indices and check bounds as an extra step.
"""
function __factorisation_specification_resolve_index end

__factorisation_specification_resolve_index(index::Any, collection::AbstractVariable)                                  = error("Attempt to access a single variable $(name(collection)) at index [$(index)].") # `index` here is guaranteed to be not `nothing`, because of dispatch. `Nothing, Nothing` version will dispatch on the method below
__factorisation_specification_resolve_index(index::Nothing, collection::AbstractVariable)                              = nothing
__factorisation_specification_resolve_index(index::Nothing, collection::AbstractVector{ <: AbstractVariable })         = nothing
__factorisation_specification_resolve_index(index::Real, collection::AbstractVector{ <: AbstractVariable })            = error("Non integer indices are not supported. Attempt to access collection $(collection) of variable $(name(first(collection))) at index [$(index)].")
__factorisation_specification_resolve_index(index::Integer, collection::AbstractVector{ <: AbstractVariable })         = (firstindex(collection) <= index <= lastindex(collection)) ? index : error("Index out of bounds happened during indices resolution in factorisation constraints. Attempt to access collection $(collection) of variable $(name(first(collection))) at index [$(index)].")
__factorisation_specification_resolve_index(index::FunctionalIndex, collection::AbstractVector{ <: AbstractVariable }) = __factorisation_specification_resolve_index(index(collection)::Integer, collection)::Integer
__factorisation_specification_resolve_index(index::CombinedRange, collection::AbstractVector{ <: AbstractVariable })   = CombinedRange(__factorisation_specification_resolve_index(firstindex(index), collection)::Integer, __factorisation_specification_resolve_index(lastindex(index), collection)::Integer)
__factorisation_specification_resolve_index(index::SplittedRange, collection::AbstractVector{ <: AbstractVariable })   = SplittedRange(__factorisation_specification_resolve_index(firstindex(index), collection)::Integer, __factorisation_specification_resolve_index(lastindex(index), collection)::Integer)

## Errors

struct ClusterIntersectionError
    expression
    varrefs
    clusters
    constraints
end

__throw_intersection_error(expression, varrefs, clusters, constraints) = throw(ClusterIntersectionError(expression, varrefs, clusters, constraints))

function Base.showerror(io::IO, error::ClusterIntersectionError)
    
    print(io, "Cluster intersection error in the expression `$(error.expression)`.\n")
    print(io, "Based on factorisation constraints the resulting local constraint ")
    print(io, "q(")
    join(io, map(r -> __io_entry_pair(r[1], r[2]), error.varrefs), ", ")
    print(io, ") = ")
    for cluster in error.clusters
        print(io, "q(")
        entries = map(cluster) do clusterindex
            __io_entry_pair(error.varrefs[clusterindex])
        end
        join(io, entries, ", ")
        print(io, ")")
    end
    print(io, " has cluster intersections, which is disallowed by default.")
    print(io, "\nTechnical info: clusters = ", error.clusters)
    print(io, "\n", error.constraints)
end