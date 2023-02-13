# This file implements various hot-fixes for external dependencies 
# This file can be empty, which is fine. It only means that all external dependecies released a new version 
# that is now fixed

# Fix for 3-argument `dot` product and `ForwardDiff.hessian`, see 
# https://github.com/JuliaDiff/ForwardDiff.jl/issues/551
# https://github.com/JuliaDiff/ForwardDiff.jl/pull/481
# https://github.com/JuliaDiff/ForwardDiff.jl/issues/480
import LinearAlgebra: dot
import ForwardDiff

function dot(x::AbstractVector, A::AbstractMatrix, y::AbstractVector{D}) where {D <: ForwardDiff.Dual}
    (axes(x)..., axes(y)...) == axes(A) || throw(DimensionMismatch())
    T = typeof(dot(first(x), first(A), first(y)))
    s = zero(T)
    i₁ = first(eachindex(x))
    x₁ = first(x)
    @inbounds for j in eachindex(y)
        yj = y[j]
        temp = zero(adjoint(A[i₁, j]) * x₁)
        @simd for i in eachindex(x)
            temp += adjoint(A[i, j]) * x[i]
        end
        s += dot(temp, yj)
    end
    return s
end
