export diageye

import LinearAlgebra: Diagonal

xT_A_x(x::AbstractVector, A::AbstractMatrix) = dot(x, A * x)

diageye(::Type{T}, n::Int) where { T <: Real } = Matrix{T}(I, n, n)
diageye(n::Int)                                = diageye(Float64, n)

function softmax(x::Array{Float64,1})

    m = maximum(x)
    map((i) -> exp(i-m), x) |> normalize_sum

end
function normalize_sum(x::Array{Float64,1}) 
    x ./ sum(x)
end
sigmoid(x) = 1/(1+exp(-x))

dtanh(x) = 1 - tanh(x)^2
