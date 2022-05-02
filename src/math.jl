export tiny, huge
export TinyNumber, HugeNumber

import Base: show, maximum
import Base: convert, promote_rule

# Tiny number

"""
    TinyNumber <: Real

`TinyNumber` represents (wow!) tiny number that can be used in a various computations without unnecessary type promotions.

See also: [`HugeNumber`](@ref)
"""
struct TinyNumber <: Real end

Base.convert(::Type{F}, ::TinyNumber) where {F <: AbstractFloat} = 10eps(F)
Base.convert(::Type{Float32}, ::TinyNumber) = 1.0f-6
Base.convert(::Type{Float64}, ::TinyNumber) = 1e-12
Base.convert(::Type{BigFloat}, ::TinyNumber) = big"1e-24"

(::Type{T})(::TinyNumber) where {T <: AbstractFloat} = convert(T, tiny)

Base.show(io::IO, ::TinyNumber) = print(io, "tiny")

Base.promote_rule(::Type{TinyNumber}, ::Type{I}) where {I <: Integer}       = promote_rule(TinyNumber, promote_type(I, Float64))
Base.promote_rule(::Type{TinyNumber}, ::Type{F}) where {F <: AbstractFloat} = F

# Huge number

"""
    HugeNumber <: Real

`HugeNumber` represents (wow!) huge number that can be used in a various computations without unnecessary type promotions.

See also: [`TinyNumber`](@ref)
"""
struct HugeNumber <: Real end

Base.convert(::Type{F}, ::HugeNumber) where {F <: AbstractFloat} = inv(convert(F, TinyNumber()))
Base.convert(::Type{Float32}, ::HugeNumber) = 1.0f+6
Base.convert(::Type{Float64}, ::HugeNumber) = 1e+12
Base.convert(::Type{BigFloat}, ::HugeNumber) = big"1e+24"

(::Type{T})(::HugeNumber) where {T <: AbstractFloat} = convert(T, huge)

Base.show(io::IO, ::HugeNumber) = print(io, "huge")

Base.promote_rule(::Type{HugeNumber}, ::Type{I}) where {I <: Integer}       = promote_rule(HugeNumber, promote_type(I, Float64))
Base.promote_rule(::Type{HugeNumber}, ::Type{F}) where {F <: AbstractFloat} = F

##

Base.promote_type(::Type{T}, ::Type{TinyNumber}, ::Type{HugeNumber}) where {T} = promote_type(promote_type(T, TinyNumber), HugeNumber)
Base.promote_type(::Type{T}, ::Type{HugeNumber}, ::Type{TinyNumber}) where {T} = promote_type(promote_type(T, HugeNumber), TinyNumber)

# 

"""
   tiny

An instance of a `TinyNumber`. Behaviour and actual value of the `tiny` number depends on the context.

# Example

```jldoctest 
julia> tiny
tiny

julia> 1 + tiny
1.000000000001

julia> tiny + 1
1.000000000001

julia> 1f0 + tiny
1.000001f0

julia> big"1.0" + tiny
1.000000000000000000000001

julia> big"1" + tiny
1.000000000000000000000001
```

See also: [`huge`](@ref), [`TinyNumber`](@ref), [`HugeNumber`](@ref)
"""
const tiny = TinyNumber()

"""
   huge

An instance of a `HugeNumber`. Behaviour and actual value of the `huge` number depends on the context.

# Example

```jldoctest 
julia> huge
huge

julia> 1 + huge
1.000000000001e12

julia> huge + 1
1.000000000001e12

julia> 1f0 + huge
1.000001f6

julia> big"1.0" + huge
1.000000000000000000000001e+24

julia> big"1" + huge
1.000000000000000000000001e+24
```

See also: [`tiny`](@ref), [`TinyNumber`](@ref), [`HugeNumber`](@ref)
"""
const huge = HugeNumber()

#

matrix_from_diagonal(diag::AbstractVector) = matrix_from_diagonal(eltype(diag), diag)

function matrix_from_diagonal(::Type{T}, diag::AbstractVector) where {T}
    size   = length(diag)
    matrix = zeros(T, (size, size))
    for i in 1:size
        matrix[i, i] = diag[i]
    end
    return matrix
end
