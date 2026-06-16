
import ExponentialFamily: InverseWishartFast

@rule MatrixNormal(:U, Marginalisation) (
    m_out::PointMass, m_M::PointMass, m_V::PointMass
) = begin
    X = mean(m_out)
    M = mean(m_M)
    V = mean(m_V)
    n, p = size(X)
    D = X - M
    Ψ = D * cholinv(V) * D'
    return InverseWishartFast(p - n - 1, Ψ)
end

# Mean-field VMP. The message in U is proportional to an (improper) InverseWishart
# with scale Ψ = E[(X-M) V⁻¹ (X-M)ᵀ]. With B = E[V⁻¹] and D = E[out] - E[M],
#   Ψ = D B Dᵀ + tr(B V_out) U_out + tr(B V_M) U_M
# (the trailing terms vanish when `out`/`M` are point masses).
@rule MatrixNormal(:U, Marginalisation) (
    q_out::Union{PointMass,MatrixNormal},
    q_M::Union{PointMass,MatrixNormal},
    q_V::Any,
) = begin
    D = mean(q_out) - mean(q_M)
    n, p = size(D)
    invV = mean(cholinv, q_V)
    Ψ = D * invV * D'
    if q_out isa MatrixNormal
        U_out, V_out = covmats(q_out)
        Ψ = Ψ + mul_trace(invV, V_out) * U_out
    end
    if q_M isa MatrixNormal
        U_M, V_M = covmats(q_M)
        Ψ = Ψ + mul_trace(invV, V_M) * U_M
    end
    return InverseWishartFast(p - n - 1, Ψ)
end
