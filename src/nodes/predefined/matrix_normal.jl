
import Distributions: MatrixNormal
import ExponentialFamily: covmats
import StatsFuns: log2π

@node MatrixNormal Stochastic [
    out, (M, aliases = [mean]), (U, aliases = [rowcov]), (V, aliases = [colcov])
]

# Average energy (mean-field) for `out ~ MatrixNormal(M, U, V)`:
#
#   U[q] = E_q[-log f]
#        = 1/2 ( p E[log|U|] + n E[log|V|] + n p log(2π) + E[tr(U⁻¹ (X-M) V⁻¹ (X-M)ᵀ)] )
#
# With a fully factorised q(out)q(M)q(U)q(V) and B = E[V⁻¹]:
#
#   E[(X-M) B (X-M)ᵀ] = D B Dᵀ + tr(B V_out) U_out + tr(B V_M) U_M,   D = E[out] - E[M]
#
# where (U_out, V_out), (U_M, V_M) are the row/column covariances of the matrix-normal
# marginals (zero when the corresponding edge is a `PointMass`), using the identity
# E[X B Xᵀ] = (E[X]) B (E[X])ᵀ + tr(B V) U for X ~ MatrixNormal(·, U, V).
@average_energy MatrixNormal (q_out::Any, q_M::Any, q_U::Any, q_V::Any) = begin
    Mout = mean(q_out)
    Mm   = mean(q_M)
    n, p = size(Mout)

    invU = mean(cholinv, q_U) # E[U⁻¹], n×n
    invV = mean(cholinv, q_V) # E[V⁻¹], p×p

    D = Mout - Mm
    Ψ = D * invV * D'

    # second-moment contributions from uncertainty in `out` and `M`
    if q_out isa MatrixNormal
        U_out, V_out = covmats(q_out)
        Ψ = Ψ + mul_trace(invV, V_out) * U_out
    end
    if q_M isa MatrixNormal
        U_M, V_M = covmats(q_M)
        Ψ = Ψ + mul_trace(invV, V_M) * U_M
    end

    result =
        p * mean(logdet, q_U) +
        n * mean(logdet, q_V) +
        n * p * log2π +
        mul_trace(invU, Ψ)
    return result / 2
end
