
import LinearAlgebra: Diagonal

function xT_A_x(x::AbstractVector, A::AbstractMatrix)
    c = zero(eltype(x))
    l = length(x)
    for i in 1:l
        @inbounds a = x[i]
        for j in 1:l
            @views @inbounds c += a * x[i] * A[i, j]
        end
    end
    return c
end

function xT_A_x(x::AbstractVector, A::Diagonal)
    c = zero(eltype(x))
    for i in 1:length(x)
        @views @inbounds c += x[i] * x[i] * A[i, i]
    end
    return c
end