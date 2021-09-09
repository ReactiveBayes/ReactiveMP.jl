export diageye, logsumexp

import LinearAlgebra: Diagonal

xT_A_x(x::AbstractVector, A::AbstractMatrix) = dot(x, A * x)

diageye(::Type{T}, n::Int) where { T <: Real } = Matrix{T}(I, n, n)
diageye(n::Int)                                = diageye(Float64, n)

function logsumexp(x)

    # calculate mean
    m = maximum(x)

    # define addmax function
    addmax(y) = y + m

    # perform log-sum-exp trick
    mapreduce((i) -> exp(i - m), +, x) |> log |> addmax
    
end