

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