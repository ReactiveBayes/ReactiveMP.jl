
import Base: merge!, show, ==, *, first, last, hash

abstract type FactorisationSpecEntryIndex end

struct FactorisationSpecEntryExact <: FactorisationSpecEntryIndex end
struct FactorisationSpecEntryIndexed <: FactorisationSpecEntryIndex end
struct FactorisationSpecEntryRanged <: FactorisationSpecEntryIndex end
struct FactorisationSpecEntrySplitRanged <: FactorisationSpecEntryIndex end

struct SplittedRange{ R <: AbstractRange }
    range :: R
end

Base.first(range::SplittedRange) = first(range.range)
Base.last(range::SplittedRange)  = last(range.range)

struct FactorisationSpecEntry{I}
    symbol :: Symbol
    index  :: I
end

name(entry::FactorisationSpecEntry) = entry.symbol

Base.show(io::IO, entry::FactorisationSpecEntry) = show(io, indextype(entry), entry)

Base.show(io, ::FactorisationSpecEntryExact, entry::FactorisationSpecEntry) = print(io, entry.symbol)
Base.show(io, ::FactorisationSpecEntryIndexed, entry::FactorisationSpecEntry) = print(io, entry.symbol, "[", entry.index, "]")
Base.show(io, ::FactorisationSpecEntryRanged, entry::FactorisationSpecEntry) = print(io, entry.symbol, "[", entry.index, "]")
Base.show(io, ::FactorisationSpecEntrySplitRanged, entry::FactorisationSpecEntry) = print(io, entry.symbol, "[", first(entry.index), "]..", entry.symbol, "[", last(entry.index), "]")

function Base.:(==)(left::FactorisationSpecEntry, right::FactorisationSpecEntry)
    return left.symbol == right.symbol && left.index == right.index
end

indextype(spec::FactorisationSpecEntry) = indextype(spec, spec.index)

indextype(::FactorisationSpecEntry, index::Nothing)       = FactorisationSpecEntryExact()
indextype(::FactorisationSpecEntry, index::Integer)       = FactorisationSpecEntryIndexed()
indextype(::FactorisationSpecEntry, index::AbstractRange) = FactorisationSpecEntryRanged()
indextype(::FactorisationSpecEntry, index::SplittedRange) = FactorisationSpecEntrySplitRanged()

function Base.merge!(left::FactorisationSpecEntry, right::FactorisationSpecEntry)
    (left.symbol === right.symbol) || error("Cannot merge factorisation specification entries with different names $(left) and $(right)")
    return merge!(indextype(left), indextype(right), left, right) 
end

function Base.merge!(::FactorisationSpecEntryIndex, ::FactorisationSpecEntryIndex, left::FactorisationSpecEntry, right::FactorisationSpecEntry)
    error("Cannot merge factorisation specification entries $(left) and $(right)")
end

function Base.merge!(::FactorisationSpecEntryExact, ::FactorisationSpecEntryExact, left::FactorisationSpecEntry, right::FactorisationSpecEntry)
    return left
end

function Base.merge!(::FactorisationSpecEntryIndexed, ::FactorisationSpecEntryIndexed, left::FactorisationSpecEntry, right::FactorisationSpecEntry)
    @assert right.index > left.index "Cannot merge factorisation specification entries $(left) and $(right). Right index should be greater than left index."
    return FactorisationSpecEntry(left.symbol, SplittedRange(left.index:right.index))
end

function Base.merge!(::FactorisationSpecEntrySplitRanged, ::FactorisationSpecEntryIndexed, left::FactorisationSpecEntry, right::FactorisationSpecEntry)
    @assert right.index > last(left.index) "Cannot merge factorisation specification entries $(left) and $(right). Right index should be greater than left index."
    return FactorisationSpecEntry(left.symbol, SplittedRange(first(left.index):right.index))
end

function Base.merge!(::FactorisationSpecEntryIndexed, ::FactorisationSpecEntrySplitRanged, left::FactorisationSpecEntry, right::FactorisationSpecEntry)
    @assert first(right.index) > left.index "Cannot merge factorisation specification entries $(left) and $(right). Right index should be greater than left index."
    return FactorisationSpecEntry(left.symbol, SplittedRange(left.index:last(right.index)))
end

function Base.merge!(::FactorisationSpecEntrySplitRanged, ::FactorisationSpecEntrySplitRanged, left::FactorisationSpecEntry, right::FactorisationSpecEntry)
    @assert first(right.index) > last(left.index) "Cannot merge factorisation specification entries $(left) and $(right). Right index should be greater than left index."
    return FactorisationSpecEntry(left.symbol, SplittedRange(first(left.index):last(right.index)))
end

# 

struct FactorisationSpec{E}
    entries :: E
end

Base.show(io::IO, spec::FactorisationSpec) = begin print(io, "q("); join(io, spec.entries, ", "); print(io, ")") end

function Base.merge!(left::FactorisationSpec, right::FactorisationSpec)
    if length(left.entries) == length(right.entries)
        if TupleTools.prod(tuple(Iterators.map((l, r) -> name(l) === name(r), left.entries, right.entries)...))
            return FactorisationSpec(tuple(Iterators.map((l, r) -> merge!(l, r), left.entries, right.entries)...))
        end
    end
    error("Cannot merge factorisation specifications $(left) and $(right)")
end

Base.merge!(left::NTuple{N, FactorisationSpec}, right::FactorisationSpec) where N = TupleTools.setindex(left, merge!(left[end], right), lastindex(left))
Base.merge!(left::FactorisationSpec, right::NTuple{N, FactorisationSpec}) where N = TupleTools.setindex(right, merge!(left, right[begin]), firstindex(right))
Base.merge!(left::NTuple{N1, FactorisationSpec}, right::NTuple{N2, FactorisationSpec}) where { N1, N2 } = TupleTools.insertat(left, lastindex(left), (merge!(left[end], right[begin]), right[begin + 1:end]...))

# Mul 

Base.:(*)(left::FactorisationSpec, right::FactorisationSpec)                    = (left, right)
Base.:(*)(left::NTuple{N, FactorisationSpec}, right::FactorisationSpec) where N = (left..., right)
Base.:(*)(left::FactorisationSpec, right::NTuple{N, FactorisationSpec}) where N = (left, right...)
Base.:(*)(left::NTuple{N1, FactorisationSpec}, right::NTuple{N2, FactorisationSpec}) where { N1, N2 } = (left..., right...)

# `Node` here refers to a node in a tree, it has nothing to do with factor nodes
struct FactorisationSpecNode{S}
    childspec :: S

    FactorisationSpecNode(childspec::S) where { N, S <: NTuple{N, FactorisationSpec} } = new{S}(childspec)
end

Base.show(io::IO, node::FactorisationSpecNode) = join(io, node.childspec, " ")

## ## 