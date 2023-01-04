export AR, Autoregressive, ARsafe, ARunsafe, ARMeta, ar_unit, ar_slice

import LazyArrays
import Distributions: VariateForm
import StatsFuns: log2π

struct AR end

const Autoregressive = AR

struct ARsafe end
struct ARunsafe end

struct ARMeta{F <: VariateForm, S}
    order::Int
    stype::S
end

function ARMeta(::Type{Univariate}, order, stype::S) where {S}
    order === 1 || @warn "ARMeta{Univariate} has been created with order equals to $(order). Order has been forced to be equal to 1."
    return ARMeta{Univariate, S}(1, stype)
end

function ARMeta(::Type{Multivariate}, order, stype::S) where {S}
    return ARMeta{Multivariate, S}(order, stype)
end

getvform(meta::ARMeta{F}) where {F} = F
getorder(meta::ARMeta)              = meta.order
getstype(meta::ARMeta)              = meta.stype

is_multivariate(meta::ARMeta) = getvform(meta) === Multivariate
is_univariate(meta::ARMeta)   = getvform(meta) === Univariate

is_safe(meta::ARMeta)   = getstype(meta) === ARsafe()
is_unsafe(meta::ARMeta) = getstype(meta) === ARunsafe()

@node AR Stochastic [y, x, θ, γ]

default_meta(::Type{AR}) = error("Autoregressive node requires meta flag explicitly specified")

@average_energy AR (q_y_x::MultivariateNormalDistributionsFamily, q_θ::NormalDistributionsFamily, q_γ::GammaShapeRate, meta::ARMeta) = begin
    mθ, Vθ   = mean_cov(q_θ)
    myx, Vyx = mean_cov(q_y_x)
    mγ       = mean(q_γ)

    order = getorder(meta)

    mx, Vx   = ar_slice(getvform(meta), myx, (order + 1):(2order)), ar_slice(getvform(meta), Vyx, (order + 1):(2order), (order + 1):(2order))
    my1, Vy1 = first(myx), first(Vyx)
    Vy1x     = ar_slice(getvform(meta), Vyx, 1, (order + 1):(2order))

    # Euivalento to AE = (-mean(log, q_γ) + log2π + mγ*(Vy1+my1^2 - 2*mθ'*(Vy1x + mx*my1) + tr(Vθ*Vx) + mx'*Vθ*mx + mθ'*(Vx + mx*mx')*mθ)) / 2
    AE = (-mean(log, q_γ) + log2π + mγ * (Vy1 + my1^2 - 2 * mθ' * (Vy1x + mx * my1) + mul_trace(Vθ, Vx) + dot(mx, Vθ, mx) + dot(mθ, Vx, mθ) + abs2(dot(mθ, mx)))) / 2

    # correction
    if is_multivariate(meta)
        # AE += entropy(q_y_x)
        idc = LazyArrays.Vcat(1, (order + 1):(2order))
        myx_n = view(myx, idc)
        Vyx_n = view(Vyx, idc, idc)
        q_y_x = MvNormalMeanCovariance(myx_n, Vyx_n)
        AE -= entropy(q_y_x)
    end

    return AE
end

@average_energy AR (q_y::NormalDistributionsFamily, q_x::NormalDistributionsFamily, q_θ::NormalDistributionsFamily, q_γ::GammaShapeRate, meta::ARMeta) = begin
    mθ, Vθ = mean_cov(q_θ)
    my, Vy = mean_cov(q_y)
    mx, Vx = mean_cov(q_x)
    mγ     = mean(q_γ)

    order = getorder(meta)

    my1, Vy1 = first(my), first(Vy)

    AE = -0.5mean(log, q_γ) + 0.5log2π + 0.5 * mγ * (Vy1 + my1^2 - 2 * mθ' * mx * my1 + mul_trace(Vθ, Vx) + dot(mx, Vθ, mx) + dot(mθ, Vx, mθ) + abs2(dot(mθ, mx)))

    # correction
    if is_multivariate(meta)
        # AE += entropy(q_y)
        q_y = NormalMeanVariance(my1, Vy1)
        AE -= entropy(q_y)
    end

    return AE
end

# Helpers for AR rules

"""
    ar_slice(::T, array, ranges...)

Returns `array[ranges...]` in case if T is Multivariate, and `first(array[ranges...])` in case if T is Univariate
"""
function ar_slice end

ar_slice(::Type{Multivariate}, array, ranges...) = view(array, ranges...)
ar_slice(::Type{Univariate}, array, ranges...) = first(view(array, ranges...))

"""
    ar_unit(::T, order)

Returns `[ 1.0, 0.0 ... 0.0 ]` with length equal to order in case if T is Multivariate, and `1.0` in case if T is Univariate
"""
function ar_unit end

ar_unit(::Type{V}, order) where {V <: VariateForm} = ar_unit(Float64, V, order)

function ar_unit(::Type{T}, ::Type{Multivariate}, order) where {T <: Real}
    return StandardBasisVector(order, 1, one(T))
    # c    = zeros(T, order)
    # c[1] = one(T)
    # return c
end

function ar_unit(::Type{T}, ::Type{Univariate}, order) where {T <: Real}
    return one(T)
end

## Allocation-free AR Precision Matrix

struct ARPrecisionMatrix{T} <: AbstractMatrix{T}
    order :: Int
    γ     :: T
end

Base.size(precision::ARPrecisionMatrix) = (precision.order, precision.order)
Base.getindex(precision::ARPrecisionMatrix, i::Int, j::Int) = (i === 1 && j === 1) ? precision.γ : ((i === j) ? convert(eltype(precision), huge) : zero(eltype(precision)))

Base.eltype(::Type{<:ARPrecisionMatrix{T}}) where {T} = T
Base.eltype(::ARPrecisionMatrix{T}) where {T}         = T

add_precision(matrix::AbstractMatrix, precision::ARPrecisionMatrix) = broadcast(+, matrix, precision)
add_precision(value::Real, precision::Real)                         = value + precision

add_precision!(matrix::AbstractMatrix, precision::ARPrecisionMatrix) = broadcast!(+, matrix, precision)
add_precision!(value::Real, precision::Real)                         = value + precision

function Base.broadcast!(::typeof(+), matrix::AbstractMatrix, precision::ARPrecisionMatrix)
    matrix[1, 1] += precision.γ
    for j in 2:first(size(matrix))
        matrix[j, j] += convert(eltype(precision), huge)
    end
    return matrix
end

ar_precision(::Type{Multivariate}, order, γ) = ARPrecisionMatrix(order, γ)
ar_precision(::Type{Univariate}, order, γ)   = γ

## Allocation-free AR Transition matrix

struct ARTransitionMatrix{T} <: AbstractMatrix{T}
    order::Int
    inv_γ::T

    function ARTransitionMatrix(order::Int, γ::T) where {T <: Real}
        return new{T}(order, inv(γ))
    end
end

Base.size(transition::ARTransitionMatrix) = (transition.order, transition.order)
Base.getindex(transition::ARTransitionMatrix, i::Int, j::Int) = (i === 1 && j === 1) ? transition.inv_γ : zero(eltype(transition))

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
