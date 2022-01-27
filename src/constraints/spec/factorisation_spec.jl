
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
getindex(entry::FactorisationSpecEntry) = entry.index

firstindex(entry::FactorisationSpecEntry) = firstindex(indextype(entry), entry)
lastindex(entry::FactorisationSpecEntry)  = lastindex(indextype(entry), entry)

firstindex(::FactorisationSpecEntryExact, entry::FactorisationSpecEntry) = typemin(Int64)
lastindex(::FactorisationSpecEntryExact, entry::FactorisationSpecEntry)  = typemax(Int64)

firstindex(::FactorisationSpecEntryIndexed, entry::FactorisationSpecEntry) = getindex(entry)
lastindex(::FactorisationSpecEntryIndexed, entry::FactorisationSpecEntry)  = getindex(entry)

firstindex(::FactorisationSpecEntryRanged, entry::FactorisationSpecEntry) = first(getindex(entry))
lastindex(::FactorisationSpecEntryRanged, entry::FactorisationSpecEntry)  = last(getindex(entry))

firstindex(::FactorisationSpecEntrySplitRanged, entry::FactorisationSpecEntry) = first(getindex(entry))
lastindex(::FactorisationSpecEntrySplitRanged, entry::FactorisationSpecEntry)  = last(getindex(entry))

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

function validate(entry::FactorisationSpecEntry, variable::AbstractVariable) 
    (name(entry) === name(variable)) || error("`validate` expects name of factorisation specification entry to be the same as the `variable` name")
    return validate(indextype(entry), entry, variable)
end

function validate(entry::FactorisationSpecEntry, variable::AbstractArray{ <: AbstractVariable }) 
    (name(entry) === name(first(variable))) || error("`validate` expects name of factorisation specification entry to be the same as the `variable` name")
    return validate(indextype(entry), entry, variable)
end

validate(::FactorisationSpecEntryExact, entry::FactorisationSpecEntry, variable::RandomVariable) = true
validate(::FactorisationSpecEntryExact, entry::FactorisationSpecEntry, variable::AbstractArray{ <: RandomVariable }) = true
validate(::FactorisationSpecEntryIndexed, entry::FactorisationSpecEntry, variable::AbstractVariable) = error("Single variable $(name(variable)) cannot be indexed in factorisation specification entry $(entry)")
validate(::FactorisationSpecEntryIndexed, entry::FactorisationSpecEntry, variable::AbstractArray{ <: RandomVariable }) = true
validate(::FactorisationSpecEntryRanged, entry::FactorisationSpecEntry, variable::AbstractVariable) = error("Single variable $(name(variable)) cannot be range-indexed in factorisation specification entry $(entry)")
validate(::FactorisationSpecEntryRanged, entry::FactorisationSpecEntry, variable::AbstractArray{ <: RandomVariable }) = (firstindex(entry) >= firstindex(variable) && lastindex(entry) <= lastindex(variable)) || error("Index out of bounds for variable $(name(first(variable))) in factorisation specification entry $(entry)")
validate(::FactorisationSpecEntrySplitRanged, entry::FactorisationSpecEntry, variable::AbstractVariable) = error("Single variable $(name(variable)) cannot be splitrange-indexed in factorisation specification entry $(entry)")
validate(::FactorisationSpecEntrySplitRanged, entry::FactorisationSpecEntry, variable::AbstractArray{ <: RandomVariable }) = (firstindex(entry) >= firstindex(variable) && lastindex(entry) <= lastindex(variable)) || error("Index out of bounds for variable $(name(first(variable))) in factorisation specification entry $(entry)")

validate(::FactorisationSpecEntryIndex, entry::FactorisationSpecEntry, variable::ConstVariable) = error("Constant $(name(variable)) should not be present in the factorisation constraints specification")
validate(::FactorisationSpecEntryIndex, entry::FactorisationSpecEntry, variable::AbstractArray{ <: ConstVariable }) = error("Constant $(name(first(variable))) should not be present in the factorisation constraints specification")
validate(::FactorisationSpecEntryIndex, entry::FactorisationSpecEntry, variable::DataVariable) = error("Data variable $(name(variable)) should not be present in the factorisation constraints specification")
validate(::FactorisationSpecEntryIndex, entry::FactorisationSpecEntry, variable::AbstractArray{ <: DataVariable }) = error("Data variable $(name(first(variable))) should not be present in the factorisation constraints specification")

# 

struct FactorisationSpec{E}
    entries :: E
end

getentries(spec::FactorisationSpec) = spec.entries

Base.show(io::IO, spec::FactorisationSpec) = begin print(io, "q("); join(io, spec.entries, ", "); print(io, ")") end

function Base.merge!(left::FactorisationSpec, right::FactorisationSpec)
    if length(left.entries) == length(right.entries)
        if TupleTools.prod(tuple(Iterators.map((l, r) -> name(l) === name(r), left.entries, right.entries)...))
            entries = tuple(Iterators.map((l, r) -> merge!(l, r), left.entries, right.entries)...)
            if length(entries) > 1
                TupleTools.prod(TupleTools.diff(map(firstindex, entries))) === 0 || error("Cannot merge factorisation specifications $(left) and $(right). First indices do not match on the left hand side of the expression and on the firght hand side.")
                TupleTools.prod(TupleTools.diff(map(lastindex, entries))) === 0 || error("Cannot merge factorisation specifications $(left) and $(right). Last indices do not match on the left hand side of the expression and on the firght hand side.")
            end
            return FactorisationSpec(entries)
        end
    end
    error("Cannot merge factorisation specifications $(left) and $(right)")
end

Base.merge!(left::NTuple{N, FactorisationSpec}, right::FactorisationSpec) where N = TupleTools.setindex(left, merge!(left[end], right), lastindex(left))
Base.merge!(left::FactorisationSpec, right::NTuple{N, FactorisationSpec}) where N = TupleTools.setindex(right, merge!(left, right[begin]), firstindex(right))
Base.merge!(left::NTuple{N1, FactorisationSpec}, right::NTuple{N2, FactorisationSpec}) where { N1, N2 } = TupleTools.insertat(left, lastindex(left), (merge!(left[end], right[begin]), right[begin + 1:end]...))

function validate(spec::FactorisationSpec, vardict, names; allow_dots = true)
    length(getentries(spec)) === length(Set(Iterators.map(name, getentries(spec)))) || error("Duplicate names in factorisation consstraint specification $(spec)")
    for entry in getentries(spec)
        if name(entry) ∉ names && !((allow_dots && name(entry) === :(..)))
            error("Factorisation specification $(spec) has unknown variable name $(name(entry))")
        elseif name(entry) !== :(..)
            if !validate(entry, vardict[ name(entry) ])
                return false
            end
        end
    end
    return true
end

# Mul 

Base.:(*)(left::FactorisationSpec, right::FactorisationSpec)                    = (left, right)
Base.:(*)(left::NTuple{N, FactorisationSpec}, right::FactorisationSpec) where N = (left..., right)
Base.:(*)(left::FactorisationSpec, right::NTuple{N, FactorisationSpec}) where N = (left, right...)
Base.:(*)(left::NTuple{N1, FactorisationSpec}, right::NTuple{N2, FactorisationSpec}) where { N1, N2 } = (left..., right...)

# `Node` here refers to a node in a tree, it has nothing to do with factor nodes
struct FactorisationSpecList{S}
    specs :: S

    FactorisationSpecList(specs::S) where { N, S <: NTuple{N, FactorisationSpec} } = new{S}(specs)
end

getentries(list::FactorisationSpecList) = list.specs

Base.show(io::IO, node::FactorisationSpecList) = join(io, node.specs, " ")

## ## 