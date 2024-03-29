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
    @assert skip >= 1
    @assert length(iterator) >= 1
    return SkipIndexIterator{eltype(I), I}(iterator, skip)
end

Base.IteratorSize(::Type{<:SkipIndexIterator})   = HasLength()
Base.IteratorEltype(::Type{<:SkipIndexIterator}) = HasEltype()
Base.IndexStyle(::Type{<:SkipIndexIterator})     = IndexLinear()

Base.eltype(::Type{<:SkipIndexIterator{T}}) where {T} = T
Base.length(iter::SkipIndexIterator)                  = length(iter.iterator) - 1
Base.size(iter::SkipIndexIterator)                    = (length(iter),)

Base.getindex(iter::SkipIndexIterator, i::Int)               = i < skip(iter) ? @inbounds(iter.iterator[i]) : @inbounds(iter.iterator[i + 1])
Base.getindex(iter::SkipIndexIterator, i::CartesianIndex{1}) = Base.getindex(iter, first(i.I))

Rocket.similar_typeof(::SkipIndexIterator, ::Type{L}) where {L} = Vector{L}

## 

import Base: +, -, *, /, convert, float, isfinite, isinf, zero, eltype

# Symbol helpers

__extract_val_type(::Type{Val{S}}) where {S} = S
__extract_val_type(::Val{S}) where {S} = S

@generated function split_underscored_symbol(symbol_val)
    S = __extract_val_type(symbol_val)
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

## 

"""
    FunctionalIndex

A special type of an index that represents a function that can be used only in pair with a collection. 
An example of a `FunctionalIndex` can be `firstindex` or `lastindex`, but more complex use cases are possible too, 
e.g. `firstindex + 1`. Important part of the implementation is that the resulting structure is `isbitstype(...) = true`, that allows to store it in parametric type as valtype.

One use case for this structure is to dispatch on and to replace `begin` or `end` (or more complex use cases, e.g. `begin + 1`) markers in constraints specification language.
"""
struct FunctionalIndex{R, F}
    f::F

    FunctionalIndex{R}(f::F) where {R, F} = new{R, F}(f)
end

(index::FunctionalIndex{R, F})(collection) where {R, F} = __functional_index_apply(R, index.f, collection)::Integer

__functional_index_apply(::Symbol, f, collection)                                               = f(collection)
__functional_index_apply(subindex::FunctionalIndex, f::Tuple{typeof(+), <:Integer}, collection) = subindex(collection) .+ f[2]
__functional_index_apply(subindex::FunctionalIndex, f::Tuple{typeof(-), <:Integer}, collection) = subindex(collection) .- f[2]

Base.:(+)(left::FunctionalIndex, index::Integer) = FunctionalIndex{left}((+, index))
Base.:(-)(left::FunctionalIndex, index::Integer) = FunctionalIndex{left}((-, index))

__functional_index_print(io::IO, f::typeof(firstindex))          = nothing
__functional_index_print(io::IO, f::typeof(lastindex))           = nothing
__functional_index_print(io::IO, f::Tuple{typeof(+), <:Integer}) = print(io, " + ", f[2])
__functional_index_print(io::IO, f::Tuple{typeof(-), <:Integer}) = print(io, " - ", f[2])

function Base.show(io::IO, index::FunctionalIndex{R, F}) where {R, F}
    print(io, "(")
    print(io, R)
    __functional_index_print(io, index.f)
    print(io, ")")
end
