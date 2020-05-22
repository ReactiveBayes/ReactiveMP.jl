export skipindex

import Base: IteratorSize, HasLength
import Base: IteratorEltype, HasEltype
import Base: eltype, length, size
import Base: IndexStyle, IndexLinear, getindex

struct SkipIndexIterator{Skip, Iterator, T, N} <: AbstractArray{T, N}
    iterator :: Iterator
end

function skipindex(iter::Iterator, skip::Int) where Iterator
    @assert skip >= 1
    @assert length(iter) >= 1
    @assert IndexStyle(Iterator) === IndexLinear()
    return SkipIndexIterator{skip, Iterator, eltype(Iterator), 1}(iter)
end

Base.IteratorSize(::Type{<:SkipIndexIterator})   = HasLength()
Base.IteratorEltype(::Type{<:SkipIndexIterator}) = HasEltype()
Base.IndexStyle(::Type{<:SkipIndexIterator})     = IndexLinear()

Base.eltype(::Type{<:SkipIndexIterator{Any, Any, T}}) where T = T
Base.length(iter::SkipIndexIterator) = max(0, length(iter.iterator) - 1)
Base.size(iter::SkipIndexIterator) = (length(iter), )

Base.getindex(iter::SkipIndexIterator{Skip}, i) where Skip = @inbounds begin i < Skip ? iter.iterator[i] : iter.iterator[i + 1] end
