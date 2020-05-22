export skipindex

import Base: IteratorSize, HasLength
import Base: IteratorEltype, HasEltype
import Base: eltype, length, size
import Base: IndexStyle, IndexLinear, getindex

struct SkipIndexIterator{Iterator, T, N} <: AbstractArray{T, N}
    iterator :: Iterator
    skip     :: Int
end

skip(iter::SkipIndexIterator) = iter.skip

function skipindex(iter::Iterator, skip::Int) where Iterator
    @assert skip >= 1
    @assert length(iter) >= 1
    @assert IndexStyle(Iterator) === IndexLinear()
    return SkipIndexIterator{Iterator, eltype(Iterator), 1}(iter, skip)
end

Base.IteratorSize(::Type{<:SkipIndexIterator})   = HasLength()
Base.IteratorEltype(::Type{<:SkipIndexIterator}) = HasEltype()
Base.IndexStyle(::Type{<:SkipIndexIterator})     = IndexLinear()

Base.eltype(::Type{<:SkipIndexIterator{Any, T}}) where T = T
Base.length(iter::SkipIndexIterator) = max(0, length(iter.iterator) - 1)
Base.size(iter::SkipIndexIterator)   = (length(iter), )

Base.getindex(iter::SkipIndexIterator, i) = @inbounds begin i < skip(iter) ? iter.iterator[i] : iter.iterator[i + 1] end
