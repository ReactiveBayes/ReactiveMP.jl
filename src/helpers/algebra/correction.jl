export NoCorrection, TinyCorrection, FixedCorrection, ClampEigenValuesCorrection

abstract type AbstractCorrection end

# Correction regularization terms for matrices

"""
    correction!(strategy, matrix)
    correction!(strategy, real)

Modifies the `matrix` with a specified correction strategy. Matrix must be squared. 
Also supports real values, with the same strategies.

See also: [`NoCorrection`](@ref), [`TinyCorrection`](@ref)
"""
function correction! end

"""
    NoCorrection

One of the correction strategies for `correction!` function. Does not modify matrix and returns an original one.

See also: [`correction!`](@ref), [`TinyCorrection`](@ref)
"""
struct NoCorrection <: AbstractCorrection end

correction!(::NoCorrection, value::Real)            = value
correction!(::NoCorrection, matrix::AbstractMatrix) = matrix
correction!(::Nothing, something)                   = correction!(NoCorrection(), something)

"""
    TinyCorrection

One of the correction strategies for `correction!` function. Adds `ReactiveMP.tiny` term to the `matrix`'s diagonal.

See also: [`correction!`](@ref), [`NoCorrection`](@ref), [`FixedCorrection`](@ref), [`ClampEigenValuesCorrection`](@ref)
"""
struct TinyCorrection <: AbstractCorrection end

correction!(::TinyCorrection, value::Real) = clamp(value, tiny, typemax(value))

function correction!(::TinyCorrection, matrix::AbstractMatrix)
    s = size(matrix)
    @assert length(s) == 2 && s[1] === s[2]
    for i in 1:s[1]
        @inbounds matrix[i, i] += tiny
    end
    return matrix
end

"""
    FixedCorrection

One of the correction strategies for `correction!` function. Adds fixed `v` term to the `matrix`'s diagonal.

# Arguments
- `v`: fixed value to add to the matrix diagonal

See also: [`correction!`](@ref), [`NoCorrection`](@ref), [`TinyCorrection`](@ref), [`ClampEigenValuesCorrection`](@ref)
"""
struct FixedCorrection{T} <: AbstractCorrection
    v::T
end

correction!(correction::FixedCorrection, value::Real) = clamp(value, correction.v, Inf)

function correction!(correction::FixedCorrection, matrix::AbstractMatrix)
    s = size(matrix)
    @assert length(s) == 2 && s[1] === s[2]
    for i in 1:s[1]
        @inbounds matrix[i, i] += correction.v
    end
    return matrix
end

"""
    ClampEigenValuesCorrection

One of the correction strategies for `correction!` function. Clamps eigen values of matrix to be equal or greater than fixed `v` term.

# Arguments
- `v`: fixed value used to clamp eigen values

See also: [`correction!`](@ref), [`NoCorrection`](@ref), [`FixedCorrection`](@ref), [`TinyCorrection`](@ref)
"""
struct ClampEigenValuesCorrection{T} <: AbstractCorrection
    v::T
end

correction!(correction::ClampEigenValuesCorrection, value::Real) = clamp(value, correction.v, Inf)

function correction!(correction::ClampEigenValuesCorrection, matrix::AbstractMatrix)
    s = size(matrix)
    @assert length(s) == 2 && s[1] === s[2]

    F = svd(matrix)
    clamp!(F.S, correction.v, Inf)
    R = lmul!(Diagonal(F.S), F.Vt)
    M = mul!(matrix, F.U, R)

    return M
end
