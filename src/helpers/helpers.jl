export skipindex

using SpecialFunctions
using Rocket

import Base: show, similar
import Base: IteratorSize, HasLength
import Base: IteratorEltype, HasEltype
import Base: eltype, length, size, sum
import Base: IndexStyle, IndexLinear, getindex

import LinearAlgebra: UniformScaling

import Rocket: similar_typeof

"""
    SkipIndexIterator

A special type of iterator that simply iterates over internal `iterator`, but skips index `skip`.

# Arguments
- `iterator`: internal iterator
- `skip`: index to skip (integer)

See also: [`skipindex`](@ref)
"""
struct SkipIndexIterator{T, I} <: AbstractVector{T}
    iterator :: I
    skip     :: Int
end

skip(iter::SkipIndexIterator) = iter.skip

"""
    skipindex(iterator, skip)

Creation operator for `SkipIndexIterator`.

```jldoctest
julia> s = ReactiveMP.skipindex(1:3, 2)
2-element ReactiveMP.SkipIndexIterator{Int64, UnitRange{Int64}}:
 1
 3

julia> collect(s)
2-element Vector{Int64}:
 1
 3
```

See also: [`SkipIndexIterator`](@ref)
"""
function skipindex(iterator::I, skip::Int) where {I}
    Base.checkbounds(Bool, iterator, skip) || throw(BoundsError(iterator, skip))
    return SkipIndexIterator{eltype(I), I}(iterator, skip)
end

function skipindex(iterator::NTuple{N}, skip::Int) where {N}
    (1 <= skip <= length(iterator)) || throw(BoundsError(iterator, skip))
    return TupleTools.deleteat(iterator, skip)
end

Base.IteratorSize(::Type{<:SkipIndexIterator})   = HasLength()
Base.IteratorEltype(::Type{<:SkipIndexIterator}) = HasEltype()
Base.IndexStyle(::Type{<:SkipIndexIterator})     = IndexLinear()

Base.length(iter::SkipIndexIterator)                  = length(iter.iterator) - 1
Base.size(iter::SkipIndexIterator)                    = (length(iter),)

Base.@propagate_inbounds Base.getindex(iter::SkipIndexIterator, i::Int)               = i < skip(iter) ? iter.iterator[i] : iter.iterator[i + 1]
Base.@propagate_inbounds Base.getindex(iter::SkipIndexIterator, i::CartesianIndex{1}) = Base.getindex(iter, first(i.I))

Rocket.similar_typeof(::SkipIndexIterator, ::Type{L}) where {L} = Vector{L}

## 

import Base: +, -, *, /, convert, float, isfinite, isinf, zero, eltype

# Symbol helpers

unval(::Type{Val{S}}) where {S} = S
unval(::Val{S}) where {S} = S

@generated function split_underscored_symbol(symbol_val)
    S = unval(symbol_val)
    R = tuple(map(Symbol, split(string(S), "_"))...)
    return :(Val{$R}())
end

# NamedTuple helpers

fields(::NamedTuple{F}) where {F} = F
hasfield(field::Symbol, ntuple::NamedTuple) = field ∈ fields(ntuple)

function swapped(tuple::Tuple, i, j)
    @assert j > i
    return (tuple[1:(i - 1)]..., tuple[j], tuple[(i + 1):(j - 1)]..., tuple[i], tuple[(j + 1):end]...)
end

function swapped(array::AbstractArray, i, j)
    array = copy(array)
    array[i], array[j] = array[j], array[i]
    return array
end

## 

__check_all(fn::Function, iterator)     = all(fn, iterator)
__check_all(fn::Function, tuple::Tuple) = TupleTools.prod(map(fn, tuple))
__check_all(fn::Function, ::Nothing)    = true

##

is_clamped_or_initial(something) = is_clamped(something) || is_initial(something)

# See: https://github.com/JuliaLang/julia/issues/42795
function fill_bitarray!(V::SubArray{Bool, <:Any, <:BitArray, <:Tuple{UnitRange{Int}}}, x)
    B = V.parent
    I0 = V.indices[1]
    l0 = length(I0)
    l0 == 0 && return V
    Base.fill_chunks!(B.chunks, Bool(x), first(I0), l0)
    return V
end

##

forward_range(range::OrdinalRange)::UnitRange = step(range) > 0 ? (first(range):last(range)) : (last(range):first(range))
