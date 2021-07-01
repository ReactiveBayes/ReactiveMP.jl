export NoCorrection, TinyCorrection

import LinearAlgebra: diagind

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

See also: [`correction!`](@ref), [`NoCorrection`](@ref)
"""
struct TinyCorrection <: AbstractCorrection end

function correction!(::TinyCorrection, matrix)
    s = size(matrix)
    @assert length(s) == 2 && s[1] === s[2]
    for i in 1:s[1]
        @inbounds matrix[i, i] = matrix[i, i] + tiny
    end
    return matrix
end
