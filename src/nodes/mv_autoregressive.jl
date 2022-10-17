export MAR, MvAutoregressive, MARMeta

import LazyArrays
import StatsFuns: log2π

struct MAR end

const MvAutoregressive = MAR

struct MARMeta
    order::Int
end

function MARMeta(order)
    return MARMeta(order)
end

getorder(meta::MARMeta)              = meta.order

@node MAR Stochastic [y, x, θ, Λ]

default_meta(::Type{MAR}) = error("MvAutoregressive node requires meta flag explicitly specified")

@average_energy AR (
    q_y_x::MultivariateNormalDistributionsFamily,
    q_θ::MultivariateNormalDistributionsFamily,
    q_Λ::Wishart,
    meta::ARMeta
) = begin
    mθ, Vθ   = mean_cov(q_θ)
    myx, Vyx = mean_cov(q_y_x)
    mΛ       = mean(q_Λ)

    order = getorder(meta)

    mx, Vx   = ar_slice(getvform(meta), myx, (order+1):2order), ar_slice(getvform(meta), Vyx, (order+1):2order, (order+1):2order)
    my1, Vy1 = first(myx), first(Vyx)
    Vy1x     = ar_slice(getvform(meta), Vyx, 1, order+1:2order)

    # Euivalento to AE = (-mean(log, q_γ) + log2π + mγ*(Vy1+my1^2 - 2*mθ'*(Vy1x + mx*my1) + tr(Vθ*Vx) + mx'*Vθ*mx + mθ'*(Vx + mx*mx')*mθ)) / 2
    AE =
        (
            -mean(log, q_γ) + log2π +
            mγ * (
                Vy1 + my1^2 - 2 * mθ' * (Vy1x + mx * my1) + mul_trace(Vθ, Vx) + dot(mx, Vθ, mx) + dot(mθ, Vx, mθ) +
                abs2(dot(mθ, mx))
            )
        ) / 2

    # correction
    if getorder(meta) > 1
        AE += entropy(q_y_x)
        idc = LazyArrays.Vcat(1, (order+1):2order)
        myx_n = view(myx, idc)
        Vyx_n = view(Vyx, idc, idc)
        q_y_x = MvNormalMeanCovariance(myx_n, Vyx_n)
        AE -= entropy(q_y_x)
    end

    return AE
end

# Helpers for AR rules

## MAllocation-free AR Precision Matrix

struct MARPrecisionMatrix{T} <: AbstractMatrix{T}
    order :: Int
    Λ     :: T
end

Base.size(precision::MARPrecisionMatrix) = (precision.order, precision.order)
Base.getindex(precision::MARPrecisionMatrix, i::Int, j::Int) =
    (i === 1 && j === 1) ? precision.Λ : ((i === j) ? convert(eltype(precision), huge) : zero(eltype(precision)))

Base.eltype(::Type{<:MARPrecisionMatrix{T}}) where {T} = T
Base.eltype(::MARPrecisionMatrix{T}) where {T}         = T

add_precision(matrix::AbstractMatrix, precision::MARPrecisionMatrix) = broadcast(+, matrix, precision)

add_precision!(matrix::AbstractMatrix, precision::MARPrecisionMatrix) = broadcast!(+, matrix, precision)

# function Base.broadcast!(::typeof(+), matrix::AbstractMatrix, precision::MARPrecisionMatrix)
#     matrix[1, 1] += precision.γ
#     for j in 2:first(size(matrix))
#         matrix[j, j] += convert(eltype(precision), huge)
#     end
#     return matrix
# end

mar_precision(::Type{Multivariate}, order, Λ) = MARPrecisionMatrix(order, Λ)
mar_precision(::Type{Univariate}, order, γ)   = γ

## Allocation-free AR Transition matrix

struct ARTransitionMatrix{T} <: AbstractMatrix{T}
    order::Int
    inv_γ::T

    function ARTransitionMatrix(order::Int, γ::T) where {T <: Real}
        return new{T}(order, inv(γ))
    end
end

Base.size(transition::ARTransitionMatrix) = (transition.order, transition.order)
Base.getindex(transition::ARTransitionMatrix, i::Int, j::Int) =
    (i === 1 && j === 1) ? transition.inv_γ : zero(eltype(transition))

Base.eltype(::Type{<:ARTransitionMatrix{T}}) where {T} = T
Base.eltype(::ARTransitionMatrix{T}) where {T}         = T

add_transition(matrix::AbstractMatrix, transition::ARTransitionMatrix) = broadcast(+, matrix, transition)
add_transition(value::Real, transition::Real)                          = value + transition

add_transition!(matrix::AbstractMatrix, transition::ARTransitionMatrix) = broadcast!(+, matrix, transition)
add_transition!(value::Real, transition::Real)                          = value + transition

function Base.broadcast!(::typeof(+), matrix::AbstractMatrix, transition::ARTransitionMatrix)
    matrix[1] += transition.inv_γ
    return matrix
end

ar_transition(::Type{Multivariate}, order, γ) = ARTransitionMatrix(order, γ)
ar_transition(::Type{Univariate}, order, γ)   = inv(γ)
