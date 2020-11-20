
import LinearAlgebra: Diagonal

xT_A_x(x::AbstractVector, A::AbstractMatrix) = dot(x, A * x)