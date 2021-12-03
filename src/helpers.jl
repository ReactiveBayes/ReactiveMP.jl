export skipindex, @symmetrical

using SpecialFunctions
using Rocket

import Base: show, similar
import Base: IteratorSize, HasLength
import Base: IteratorEltype, HasEltype
import Base: eltype, length, size, sum
import Base: IndexStyle, IndexLinear, getindex

import Rocket: similar_typeof

struct SkipIndexIterator{T, I} <: AbstractVector{T}
    iterator :: I
    skip     :: Int
end

skip(iter::SkipIndexIterator) = iter.skip

function skipindex(iterator::I, skip::Int) where I
    @assert skip >= 1
    @assert length(iterator) >= 1
    return SkipIndexIterator{eltype(I), I}(iterator, skip)
end

Base.IteratorSize(::Type{<:SkipIndexIterator})   = HasLength()
Base.IteratorEltype(::Type{<:SkipIndexIterator}) = HasEltype()
Base.IndexStyle(::Type{<:SkipIndexIterator})     = IndexLinear()

Base.eltype(::Type{<:SkipIndexIterator{T}}) where T = T
Base.length(iter::SkipIndexIterator)                = length(iter.iterator) - 1
Base.size(iter::SkipIndexIterator)                  = (length(iter), )

Base.getindex(iter::SkipIndexIterator, i::Int)               = i < skip(iter) ? @inbounds(iter.iterator[i]) : @inbounds(iter.iterator[i + 1])
Base.getindex(iter::SkipIndexIterator, i::CartesianIndex{1}) = Base.getindex(iter, first(i.I))

Rocket.similar_typeof(::SkipIndexIterator, ::Type{L}) where L = Vector{L}

reduce_with_sum(array) = reduce(+, array)

## 

import Base: +, -, *, /, convert, float, isfinite, isinf, zero, eltype

struct InfCountingReal{ T <: Real }
    value :: T
    infs  :: Int
end

InfCountingReal(value::T)            where { T <: Real } = InfCountingReal{T}(value, 0)
InfCountingReal(::Type{T}, inf::Int) where { T <: Real } = InfCountingReal{T}(zero(T), inf)

Infinity(::Type{T}) where { T <: Real } = InfCountingReal(T, 1)

const ∞ = Infinity(Float64)

value(a::InfCountingReal) = a.value
infs(a::InfCountingReal)  = a.infs

Base.isfinite(a::InfCountingReal) = infs(a) === 0
Base.isinf(a::InfCountingReal)    = !(isfinite(a))

Base.eltype(::Type{ InfCountingReal{T} }) where T = T
Base.eltype(::Type{ InfCountingReal })            = Real

Base.eltype(::T) where { T <: InfCountingReal } = eltype(T)

Base.:+(a::InfCountingReal) = InfCountingReal(+value(a), +infs(a))
Base.:-(a::InfCountingReal) = InfCountingReal(-value(a), -infs(a))

Base.:+(a::InfCountingReal, b::Real) = InfCountingReal(value(a) + b, infs(a))
Base.:-(a::InfCountingReal, b::Real) = InfCountingReal(value(a) - b, infs(a))
Base.:+(b::Real, a::InfCountingReal) = InfCountingReal(b + value(a), +infs(a))
Base.:-(b::Real, a::InfCountingReal) = InfCountingReal(b - value(a), -infs(a))

Base.:*(::InfCountingReal, ::Real) = error("InfCountingReal multiplication is dissalowed")
Base.:/(::InfCountingReal, ::Real) = error("InfCountingReal division is dissalowed")
Base.:*(::Real, ::InfCountingReal) = error("InfCountingReal multiplication is dissalowed")
Base.:/(::Real, ::InfCountingReal) = error("InfCountingReal division is dissalowed")

Base.:+(a::InfCountingReal, b::InfCountingReal) = InfCountingReal(value(a) + value(b), infs(a) + infs(b))
Base.:-(a::InfCountingReal, b::InfCountingReal) = InfCountingReal(value(a) - value(b), infs(a) - infs(b))

Base.convert(::Type{ InfCountingReal },    v::T)                  where { T <: Real }            = InfCountingReal(v)
Base.convert(::Type{ InfCountingReal{T} }, v::T)                  where { T <: Real }            = InfCountingReal(v)
Base.convert(::Type{ InfCountingReal{T} }, v::R)                  where { T <: Real, R <: Real } = InfCountingReal(convert(T, v))
Base.convert(::Type{ InfCountingReal{T} }, v::InfCountingReal{R}) where { T <: Real, R <: Real } = InfCountingReal{T}(convert(T, value(v)), infs(v))

Base.float(a::InfCountingReal) = isfinite(a) ? value(a) : Inf

Base.zero(::Type{InfCountingReal{T}}) where { T <: Real } = InfCountingReal(zero(T))

Base.show(io::IO, a::InfCountingReal{T}) where T = print(io, "InfCountingReal($(value(a)), $(infs(a))∞)")

Base.promote_rule(::Type{ InfCountingReal{T1} }, ::Type{ T2 }) where { T1 <: Real, T2 <: Real } = InfCountingReal{ promote_type(T1, T2) }
Base.promote_rule(::Type{ InfCountingReal },     ::Type{ T })  where { T <: Real }              = InfCountingReal{ T }

Base.:(==)(left::InfCountingReal{T}, right::InfCountingReal{T}) where { T } = (left.value == right.value) && (left.infs == right.infs)

# Union helpers

union_types(x::Union) = (x.a, union_types(x.b)...)
union_types(x::Type)  = (x,)

# Symbol helpers

__extract_val_type(::Type{ Type{ Val{ S } } }) where S = S
__extract_val_type(::Type{ Val{ S } })         where S = S

@generated function split_underscored_symbol(symbol_val)
    S = __extract_val_type(symbol_val)
    R = tuple(map(Symbol, split(string(S), "_"))...)
    return :(Val{ $R }) 
end

# NamedTuple helpers

fields(::NamedTuple{ F }) where F  = F
hasfield(field::Symbol, ntuple::NamedTuple) = field ∈ fields(ntuple)

function swapped(tuple::Tuple, i, j)
    @assert j > i 
    return (tuple[1:i-1]..., tuple[j], tuple[i+1:j-1]..., tuple[i], tuple[j+1:end]...)
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

## Meta utils

"""
    @symmetrical `function_definition`
Duplicate a method definition with the order of the first two arguments swapped.
This macro is used to duplicate methods that are symmetrical in their first two input arguments,
but require explicit definitions for the different argument orders.
Example:
    @symmetrical function prod!(x, y, z)
        ...
    end
"""
macro symmetrical(fn::Expr)
    # Check if macro is applied to a function definition
    # Valid function definitions include:
    # 1. foo([ args... ]) [ where ... [ where ... [ ... ] ] ] = :block
    # 2. function foo([ args... ]) [ where ... [ where ... [ ... ] ] ]
    #        :block
    #    end
    if (fn.head === :(=) || fn.head === :function) &&
        (fn.args[1] isa Expr && fn.args[2] isa Expr) &&
        (fn.args[2].head === :block)
        return esc(quote
            $fn
            $(swap_arguments(fn))
        end)
    else
        error("@symmetrical macro can be applied only to function definitions")
    end
end

function swap_arguments(fn::Expr)
    swapped = copy(fn)

    if swapped.args[1].head === :where
        swapped.args[1] = swap_arguments(swapped.args[1])
    elseif swapped.args[1].head === :call && length(fn.args[1].args) >= 3 # Note: >= 3, because the first argument is a function name
        swapped.args[1].args[2] = fn.args[1].args[3]
        swapped.args[1].args[3] = fn.args[1].args[2]
    else
        error("Function method passed for @symmetrical macro must have more than 2 arguments")
    end

    return swapped
end

## Other helpers 

# We override this function for some specific types
function is_typeof_equal(left, right) 
    _isequal = typeof(left) === typeof(right)
    if !_isequal
        @warn "typeof($left) !== typeof($right)"
    end
    return _isequal
end

function custom_isapprox(left, right; kwargs...)
    _isapprox = isapprox(left, right; kwargs...)
    if !_isapprox
        @warn "$left !≈ $right" 
    end
    return _isapprox
end

custom_isapprox(left::NamedTuple, right::NamedTuple; kwargs...) = false

function custom_isapprox(left::NamedTuple{K}, right::NamedTuple{K}; kwargs...) where { K } 
    _isapprox = true
    for key in keys(left)
        _isapprox = _isapprox && custom_isapprox(left[key], right[key]; kwargs...)
    end
    if !_isapprox
        @warn "$left !≈ $right" 
    end
    return _isapprox
end

## 

deep_eltype(::Type{ T })  where { T <: Number } = T
deep_eltype(::Type{ T })  where T               = deep_eltype(eltype(T))
deep_eltype(::T)          where T               = deep_eltype(T)    

##

# See: https://github.com/JuliaLang/julia/issues/42795
function fill_bitarray!(V::SubArray{Bool, <:Any, <:BitArray, <:Tuple{AbstractUnitRange{Int}}}, x)
    B = V.parent
    I0 = V.indices[1]
    l0 = length(I0)
    l0 == 0 && return V
    Base.fill_chunks!(B.chunks, Bool(x), first(I0), l0)
    return V
end

## 

default_if_nothing(::Nothing, default) = default
default_if_nothing(any::Any, default)  = any

##

forward_range(range::OrdinalRange) = step(range) > 0 ? range : last(range):first(range)