"""
    CountingReal

`CountingReal` implements a "number" that counts 'improper' real values in a separate field. Used to cancel out infinities in BFE computations.
`Improper` implemented as `CountingReal(zero(Float64), 1)`.

# Arguments
- `value::T`: value of type `<: Real`
- `improper::Int`: number of added/subtracted improper real values

```jldoctest
julia> r = ReactiveMP.CountingReal(0.0, 0)
CountingReal(0.0, 0)

julia> float(r)
0.0

julia> r = r + ReactiveMP.ImproperReal()
CountingReal(0.0, 1)

julia> float(r)
Inf

julia> r = r - ReactiveMP.ImproperReal()
CountingReal(0.0, 0)

julia> float(r)
0.0
```

See also: [`ReactiveMP.ImproperReal`](@ref)
"""
struct CountingReal{T <: Real}
    value    :: T
    improper :: Int
end

CountingReal(value::T) where {T <: Real}                 = CountingReal{T}(value, 0)
CountingReal(::Type{T}, improper::Int) where {T <: Real} = CountingReal{T}(zero(T), improper)

ImproperReal()                            = ImproperReal(Float64)
ImproperReal(::Type{T}) where {T <: Real} = CountingReal(T, 1)

value(a::CountingReal)    = a.value
improper(a::CountingReal) = a.improper

value_isnan(a) = isnan(a)
value_isinf(a) = isinf(a)
value_isnan(a::CountingReal) = isnan(value(a))
value_isinf(a::CountingReal) = isinf(value(a))

Base.isfinite(a::CountingReal) = improper(a) === 0
Base.isinf(a::CountingReal)    = !(isfinite(a))

Base.eltype(::Type{CountingReal{T}}) where {T} = T
Base.eltype(::Type{CountingReal})              = Real

Base.eltype(::T) where {T <: CountingReal} = eltype(T)

Base.:+(a::CountingReal) = CountingReal(+value(a), +improper(a))
Base.:-(a::CountingReal) = CountingReal(-value(a), -improper(a))

Base.:+(a::CountingReal, b::Real) = CountingReal(value(a) + b, improper(a))
Base.:-(a::CountingReal, b::Real) = CountingReal(value(a) - b, improper(a))
Base.:+(b::Real, a::CountingReal) = CountingReal(b + value(a), +improper(a))
Base.:-(b::Real, a::CountingReal) = CountingReal(b - value(a), -improper(a))

Base.:*(::CountingReal, ::Real) = error("`CountingReal` multiplication is dissalowed")
Base.:/(::CountingReal, ::Real) = error("`CountingReal` division is dissalowed")
Base.:*(::Real, ::CountingReal) = error("`CountingReal` multiplication is dissalowed")
Base.:/(::Real, ::CountingReal) = error("`CountingReal` division is dissalowed")

Base.:+(a::CountingReal, b::CountingReal) = CountingReal(value(a) + value(b), improper(a) + improper(b))
Base.:-(a::CountingReal, b::CountingReal) = CountingReal(value(a) - value(b), improper(a) - improper(b))

Base.convert(::Type{CountingReal}, v::T) where {T <: Real}                             = CountingReal(v)
Base.convert(::Type{CountingReal{T}}, v::T) where {T <: Real}                          = CountingReal(v)
Base.convert(::Type{CountingReal{T}}, v::R) where {T <: Real, R <: Real}               = CountingReal(convert(T, v))
Base.convert(::Type{CountingReal{T}}, v::CountingReal{R}) where {T <: Real, R <: Real} = CountingReal{T}(convert(T, value(v)), improper(v))

Base.float(a::CountingReal) = isfinite(a) ? value(a) : Inf

Base.zero(::Type{CountingReal{T}}) where {T <: Real} = CountingReal(zero(T))

Base.show(io::IO, a::CountingReal{T}) where {T} = print(io, "CountingReal($(value(a)), $(improper(a)))")

Base.promote_rule(::Type{CountingReal{T1}}, ::Type{T2}) where {T1 <: Real, T2 <: Real} = CountingReal{promote_type(T1, T2)}
Base.promote_rule(::Type{CountingReal}, ::Type{T}) where {T <: Real}                   = CountingReal{T}

Base.:(==)(left::CountingReal{T}, right::CountingReal{T}) where {T} = (left.value == right.value) && (left.improper == right.improper)