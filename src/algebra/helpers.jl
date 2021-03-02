
import LinearAlgebra: Diagonal

xT_A_x(x::AbstractVector, A::AbstractMatrix) = dot(x, A * x)

# AR related
function wMatrix(γ, order, mtype::Type{Multivariate})
    mW = huge * Matrix{Float64}(I, order, order)
    mW[1, 1] = γ
    return mW
end

wMatrix(γ, order, mtype::Type{Univariate}) = γ

function transition(γ, order, mtype::Type{Multivariate})
    V = zeros(order, order)
    V[1] = 1/γ
    return V
end

transition(γ, order, mtype::Type{Univariate}) = 1/γ
