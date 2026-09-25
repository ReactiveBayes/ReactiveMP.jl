@define_message_update_rule(
    node = MatrixNormal, target = :M,
    args = (m[:out]::PointMass, m[:U]::PointMass, m[:V]::PointMass),
    logscale = 0,
    body = (args) -> MatrixNormal(mean(args.m[:out]), mean(args.m[:U]), mean(args.m[:V])),
)

# An uncertain `out` makes the covariance of vec(M) V ⊗ U + Cov[vec(out)], which is not a
# Kronecker product, so the message is a vectorised normal rather than a MatrixNormal.
@define_message_update_rule(
    node = MatrixNormal, target = :M,
    args = (m[:out]::MatrixNormal, m[:U]::PointMass, m[:V]::PointMass),
    body = (args) -> MvNormalMeanCovariance(vec(mean(args.m[:out])), kron(mean(args.m[:V]), mean(args.m[:U])) + cov(args.m[:out])),
)

# An InverseWishart U or V integrates out into a matrix-t: X | U ~ MatrixNormal(M, U, V) with
# U ~ InverseWishart(ν, Ψ) is MatrixTDist(ν - n + 1, M, Ψ, V), and the same for V with p.
@define_message_update_rule(
    node = MatrixNormal, target = :M,
    args = (m[:out]::PointMass, m[:U]::InverseWishartDistributionsFamily, m[:V]::PointMass),
    body = (args) -> begin
        X = mean(args.m[:out])
        ν, Ψ = params(args.m[:U])
        MatrixTDist(ν - size(X, 1) + 1, X, Ψ, mean(args.m[:V]))
    end,
)

@define_message_update_rule(
    node = MatrixNormal, target = :M,
    args = (m[:out]::PointMass, m[:U]::PointMass, m[:V]::InverseWishartDistributionsFamily),
    body = (args) -> begin
        X = mean(args.m[:out])
        ν, Ψ = params(args.m[:V])
        MatrixTDist(ν - size(X, 2) + 1, X, mean(args.m[:U]), Ψ)
    end,
)

# Mean field: MatrixNormal(E[out], E[U⁻¹]⁻¹, E[V⁻¹]⁻¹), symmetrised. A MatrixNormal
# `q_out` and a point mass are both accepted.
@define_message_update_rule(
    node = MatrixNormal, target = :M,
    args = (q[:out]::PointOrMatrixNormal, q[:U]::Union{InverseWishartDistributionsFamily, PointMass}, q[:V]::Union{InverseWishartDistributionsFamily, PointMass}),
    body = (args) -> begin
        U = cholinv(mean(cholinv, args.q[:U]))
        V = cholinv(mean(cholinv, args.q[:V]))
        MatrixNormal(mean(args.q[:out]), (U + U') / 2, (V + V') / 2)
    end,
)
