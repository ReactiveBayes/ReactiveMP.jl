export skipindex, @symmetrical

using SpecialFunctions
using Rocket

import Base: show
import Base: IteratorSize, HasLength
import Base: IteratorEltype, HasEltype
import Base: eltype, length, size
import Base: IndexStyle, IndexLinear, getindex

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

Base.getindex(iter::SkipIndexIterator, i) = i < skip(iter) ? @inbounds(iter.iterator[i]) : @inbounds(iter.iterator[i + 1])

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

cast_to_message_subscribable(some::T) where T = cast_to_message_subscribable(as_subscribable(T), some)

cast_to_message_subscribable(::InvalidSubscribableTrait, some)   = of(as_message(some))
cast_to_message_subscribable(::SimpleSubscribableTrait, some)    = some |> map(Message, as_message)
cast_to_message_subscribable(::ScheduledSubscribableTrait, some) = some |> map(Message, as_message)

reduce_with_sum(array) = reduce(+, array)

## 

import Base: +, -, *, /, convert, float, isfinite, isinf, zero, eltype

struct Infinity 
    degree :: Int
end

degree(a::Infinity) = a.degree

const ∞ = Infinity(1)

Base.:+(a::Infinity, b::Infinity) = Infinity(degree(a) + degree(b))
Base.:-(a::Infinity, b::Infinity) = Infinity(degree(a) - degree(b))
Base.:+(a::Infinity)              = Infinity(+degree(a))
Base.:-(a::Infinity)              = Infinity(-degree(a))

Base.:*(::Infinity, ::Real)     = error("Infinity multiplication is disallowed")
Base.:*(::Real, ::Infinity)     = error("Infinity multiplication is disallowed")
Base.:*(::Infinity, ::Infinity) = error("Infinity multiplication is disallowed")
Base.:/(::Infinity, ::Real)     = error("Infinity division is disallowed")
Base.:/(::Real, ::Infinity)     = error("Infinity division is disallowed")
Base.:/(::Infinity, ::Infinity) = error("Infinity division is disallowed")

Base.zero(::Type{Infinity})       = Infinity(0)

Base.show(io::IO, a::Infinity) = print(io, "Infinity($(degree(a)))")

struct InfCountingReal{ T <: Real }
    value :: T
    infs  :: Int
end

InfCountingReal(value::T)                 where { T <: Real } = InfCountingReal{T}(value, 0)
InfCountingReal(::Type{T}, inf::Infinity) where { T <: Real } = InfCountingReal{T}(zero(T), degree(inf))

value(a::InfCountingReal) = a.value
infs(a::InfCountingReal)  = a.infs

Base.isfinite(a::InfCountingReal) = infs(a) === 0
Base.isinf(a::InfCountingReal)    = !(isfinite(a))

Base.eltype(::Type{ <: InfCountingReal{T} }) where T = T
Base.eltype(::Type{ <: InfCountingReal })            = Real

Base.eltype(::T) where { T <: InfCountingReal } = eltype(T)

Base.:+(a::Infinity, b::Real) = InfCountingReal(b, degree(a))
Base.:-(a::Infinity, b::Real) = InfCountingReal(-b, degree(a))
Base.:+(b::Real, a::Infinity) = InfCountingReal(b, +degree(a))
Base.:-(b::Real, a::Infinity) = InfCountingReal(b, -degree(a))

Base.:+(a::InfCountingReal) = InfCountingReal(+value(a), +infs(a))
Base.:-(a::InfCountingReal) = InfCountingReal(-value(a), -infs(a))

Base.:+(a::InfCountingReal, b::Infinity) = InfCountingReal(value(a), infs(a) + degree(b))
Base.:-(a::InfCountingReal, b::Infinity) = InfCountingReal(value(a), infs(a) - degree(b))

Base.:+(b::Infinity, a::InfCountingReal) = InfCountingReal(+value(a), degree(b) + infs(a))
Base.:-(b::Infinity, a::InfCountingReal) = InfCountingReal(-value(a), degree(b) - infs(a))

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
Base.convert(::Type{ InfCountingReal{T} }, inf::Infinity)         where { T <: Real }            = InfCountingReal(T, inf)
Base.convert(::Type{ InfCountingReal{T} }, v::InfCountingReal{R}) where { T <: Real, R <: Real } = InfCountingReal{T}(convert(T, value(v)), infs(v))

Base.float(a::InfCountingReal) = isfinite(a) ? value(a) : Inf

Base.zero(::Type{InfCountingReal{T}}) where { T <: Real } = InfCountingReal(zero(T))

Base.show(io::IO, a::InfCountingReal{T}) where T = print(io, "InfCountingReal($(value(a)), $(infs(a))∞)")

# Union helpers

union_types(x::Union) = (x.a, union_types(x.b)...)
union_types(x::Type)  = (x,)

# Symbol helpers

__extract_val_type(::Type{ Type{ Val{ S } } }) where S = S

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

