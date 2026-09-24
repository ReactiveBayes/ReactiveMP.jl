# The AR state's noise, allocation-free: only its first component is noisy, with precision γ.
#
# ARTransitionMatrix is the covariance, inv(γ) at [1, 1] and zero elsewhere; ARPrecisionMatrix
# the precision, γ at [1, 1] and `huge` on the rest of the diagonal, a regularised inverse of
# it. Both are internal. `add_transition` and `add_precision` add one to a matrix, into a new
# one or, with `!`, in place; for a univariate AR(1) they are scalars and the sum is plain.
# v6 overloaded `broadcast!(+, matrix, noise)` to mean the in-place sum, which is not what
# broadcasting means (`broadcast!(+, dest, A)` sets `dest` to `A`); the `!` functions do it
# instead, and `broadcast` itself is the generic one.

struct ARTransitionMatrix{T <: Real} <: AbstractMatrix{T}
    order::Int
    inv_γ::T

    ARTransitionMatrix{T}(order::Int, inv_γ) where {T <: Real} = new{T}(order, inv_γ)
end

# Constructed from the precision γ; the type parameter's constructor takes inv(γ) itself.
ARTransitionMatrix(order::Int, γ::Real) = ARTransitionMatrix{typeof(inv(γ))}(order, inv(γ))

Base.size(transition::ARTransitionMatrix) = (transition.order, transition.order)

function Base.getindex(transition::ARTransitionMatrix, i::Int, j::Int)
    @boundscheck checkbounds(transition, i, j)
    return (i == 1 && j == 1) ? transition.inv_γ : zero(eltype(transition))
end

Base.convert(::Type{AbstractArray{T}}, matrix::ARTransitionMatrix) where {T <: Real} =
    ARTransitionMatrix{T}(matrix.order, convert(T, matrix.inv_γ))

struct ARPrecisionMatrix{T <: Real} <: AbstractMatrix{T}
    order::Int
    γ::T

    ARPrecisionMatrix{T}(order::Int, γ) where {T <: Real} = new{T}(order, γ)
end

ARPrecisionMatrix(order::Int, γ::T) where {T <: Real} = ARPrecisionMatrix{T}(order, γ)

Base.size(precision::ARPrecisionMatrix) = (precision.order, precision.order)

function Base.getindex(precision::ARPrecisionMatrix, i::Int, j::Int)
    @boundscheck checkbounds(precision, i, j)
    (i == 1 && j == 1) && return precision.γ
    return i == j ? convert(eltype(precision), huge) : zero(eltype(precision))
end

Base.convert(::Type{AbstractArray{T}}, matrix::ARPrecisionMatrix) where {T <: Real} =
    ARPrecisionMatrix{T}(matrix.order, convert(T, matrix.γ))

function check_noise_size(matrix, noise)
    size(matrix) == size(noise) || throw(DimensionMismatch("cannot add a noise matrix of size $(size(noise)) to one of size $(size(matrix))"))
    return nothing
end

add_transition(matrix::AbstractMatrix, transition::ARTransitionMatrix) = broadcast(+, matrix, transition)
add_transition(value::Real, transition::Real) = value + transition

function add_transition!(matrix::AbstractMatrix, transition::ARTransitionMatrix)
    check_noise_size(matrix, transition)
    matrix[1, 1] += transition.inv_γ
    return matrix
end
add_transition!(value::Real, transition::Real) = value + transition

add_precision(matrix::AbstractMatrix, precision::ARPrecisionMatrix) = broadcast(+, matrix, precision)
add_precision(value::Real, precision::Real) = value + precision

function add_precision!(matrix::AbstractMatrix, precision::ARPrecisionMatrix)
    check_noise_size(matrix, precision)
    matrix[1, 1] += precision.γ
    for j in 2:first(size(matrix))
        matrix[j, j] += convert(eltype(precision), huge)
    end
    return matrix
end
add_precision!(value::Real, precision::Real) = value + precision

ar_transition(::Type{Multivariate}, order, γ) = ARTransitionMatrix(order, γ)
ar_transition(::Type{Univariate}, order, γ) = inv(γ)

ar_precision(::Type{Multivariate}, order, γ) = ARPrecisionMatrix(order, γ)
ar_precision(::Type{Univariate}, order, γ) = γ
