export diageye

using StatsFuns: logistic

import LinearAlgebra: Diagonal

xT_A_x(x::AbstractVector, A::AbstractMatrix) = dot(x, A * x)

diageye(::Type{T}, n::Int) where { T <: Real } = Matrix{T}(I, n, n)
diageye(n::Int)                                = diageye(Float64, n)

function normalize_sum(x::Array{Float64,1}) 
    x ./ sum(x)
end
const sigmoid = logistic

dtanh(x) = 1 - tanh(x)^2
