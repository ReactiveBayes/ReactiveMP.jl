export NoCorrection, TinyCorrection, FixedCorrection, ClampEigenValuesCorrection

abstract type AbstractCorrection end

# Correction regularization terms for matrices

"""
    correction!(strategy, matrix)

Modifies the `matrix` with a specified correction strategy. Matrix must be squared.
See also: [`NoCorrection`](@ref), [`TinyCorrection`](@ref)
"""
function correction! end

"""
    NoCorrection

One of the correction strategies for `correction!` function. Does not modify matrix and returns an original one.

See also: [`correction!`](@ref), [`TinyCorrection`](@ref)
"""
struct NoCorrection <: AbstractCorrection end

function correction!(::NoCorrection, matrix)
    return matrix
end


"""
    TinyCorrection

One of the correction strategies for `correction!` function. Adds `ReactiveMP.tiny` term to the `matrix`'s diagonal.

See also: [`correction!`](@ref), [`NoCorrection`](@ref), [`FixedCorrection`](@ref), [`ClampEigenValuesCorrection`](@ref)
"""
struct TinyCorrection <: AbstractCorrection end

function correction!(::TinyCorrection, matrix)
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
    v :: T
end

function correction!(correction::FixedCorrection, matrix)
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
    v :: T
end

function correction!(correction::ClampEigenValuesCorrection, matrix)
    s = size(matrix)
    @assert length(s) == 2 && s[1] === s[2]

    F = svd(matrix)
    clamp!(F.S, correction.v, Inf)
    R = lmul!(Diagonal(F.S), F.Vt)
    M = mul!(matrix, F.U, R)

    return M
end

