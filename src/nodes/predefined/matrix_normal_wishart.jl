
import ExponentialFamily: MatrixNormalWishart
import SpecialFunctions: loggamma, digamma
import StatsFuns: log2π, logπ

@node MatrixNormalWishart Stochastic [
    out, (M, aliases = [mean]), (U, aliases = [rowcov]), (V, aliases = [scale]), (ν, aliases = [dof])
]

# Average energy (conjugate-prior node) for `out = (X, Y) ~ MatrixNormalWishart(M, U, V, ν)`,
# where `X | Y ~ MatrixNormal(M, U, Y⁻¹)` (Y is the column precision) and `Y ~ Wishart(ν, V)`.
#
# With the joint belief `q_out ~ MatrixNormalWishart(Mq, Uq, Vq, νq)` (so the Y-marginal is
# `Wishart(νq, Vq)`) and point-mass hyperparameters M, U, V, ν, the average energy
# `E_q[-log f] = E[-log MatrixNormal(X; M, U, Y⁻¹)] + E[-log Wishart(Y; ν, V)]` is
#
#   U[q] = (n p /2) log2π + (p/2) logdet(U) + (ν p /2) log2 + (ν/2) logdet(V) + logΓ_p(ν/2)
#        − ((n + ν − p − 1)/2) L
#        + 1/2 ( tr(U⁻¹ D (νq Vq) Dᵀ) + p tr(U⁻¹ Uq) + νq tr(V⁻¹ Vq) ),   D = Mq − M
#
# using E[Y] = νq Vq, L = E[log|Y|], and the matrix-normal identity
# E_{X|Y}[(X−M) Y (X−M)ᵀ] = D Y Dᵀ + tr(Y Y⁻¹) Uq = D Y Dᵀ + p Uq.

@average_energy MatrixNormalWishart (
    q_out::MatrixNormalWishart, q_M::Any, q_U::Any, q_V::Any, q_ν::Any
) = begin
    Mq, Uq, Vq, νq = params(q_out)
    M = mean(q_M)
    U = mean(q_U)
    V = mean(q_V)
    ν = mean(q_ν)
    n, p = size(Mq)

    invU = cholinv(U)
    invV = cholinv(V)

    # E[log|Y|] for Y ~ Wishart(νq, Vq)
    L = p * log(2) + logdet(Vq) + mapreduce(i -> digamma((νq + 1 - i) / 2), +, 1:p)
    
    # log Γ_p(ν/2)
    logΓp = (p * (p - 1) / 4) * logπ + mapreduce(i -> loggamma((ν + 1 - i) / 2), +, 1:p)

    D = Mq - M
    quad = mul_trace(invU, D * (νq * Vq) * D') + p * mul_trace(invU, Uq)
    trVY = νq * mul_trace(invV, Vq)

    return (n * p / 2) * log2π +
        (p / 2) * logdet(U) +
        (ν * p / 2) * log(2) +
        (ν / 2) * logdet(V) +
        logΓp -
        ((n + ν - p - 1) / 2) * L +
        (quad + trVY) / 2
end
