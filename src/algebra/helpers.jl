export diageye

import LinearAlgebra: Diagonal

xT_A_x(x::AbstractVector, A::AbstractMatrix) = dot(x, A, x)

# TODO: Why not just use `I(n)`? The current implementation forces allocation,
# where it wouldn't seem to be needed
diageye(::Type{T}, n::Int) where { T <: Real } = Matrix{T}(I, n, n)
diageye(n::Int)                                = diageye(Float64, n)