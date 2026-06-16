
import ExponentialFamily: InverseWishartFast

@rule MatrixNormal(:V, Marginalisation) (
    m_out::PointMass, m_M::PointMass, m_U::PointMass
) = begin
    X = mean(m_out)
    M = mean(m_M)
    U = mean(m_U)
    n, p = size(X)
    D = X - M
    Ψ = D' * cholinv(U) * D
    return InverseWishartFast(n - p - 1, Ψ)
end

# Mean-field VMP. The message in V is proportional to an (improper) InverseWishart
# with scale Ψ = E[(X-M)ᵀ U⁻¹ (X-M)]. With A = E[U⁻¹] and D = E[out] - E[M],
#   Ψ = Dᵀ A D + tr(A U_out) V_out + tr(A U_M) V_M
# (the trailing terms vanish when `out`/`M` are point masses).
@rule MatrixNormal(:V, Marginalisation) (
    q_out::Union{PointMass, MatrixNormal},
    q_M::Union{PointMass, MatrixNormal},
    q_U::Any,
) = begin
    D = mean(q_out) - mean(q_M)
    n, p = size(D)
    invU = mean(cholinv, q_U)
    Ψ = D' * invU * D
    if q_out isa MatrixNormal
        U_out, V_out = covmats(q_out)
        Ψ = Ψ + mul_trace(invU, U_out) * V_out
    end
    if q_M isa MatrixNormal
        U_M, V_M = covmats(q_M)
        Ψ = Ψ + mul_trace(invU, U_M) * V_M
    end
    return InverseWishartFast(n - p - 1, Ψ)
end
