export diageye

import LinearAlgebra: Diagonal

xT_A_x(x::AbstractVector, A::AbstractMatrix) = dot(x, A * x)

diageye(::Type{T}, n::Int) where { T <: Real } = Matrix{T}(I, n, n)
diageye(n::Int)                                = diageye(Float64, n)

negate_inplace!(A::AbstractMatrix) = map!(-, A, A)
negate_inplace!(A::Real)           = -A