# The likelihood of U: |U|^(-p/2) exp(-tr(U⁻¹ Ψ)/2) with Ψ = E[(out - M) E[V⁻¹] (out - M)ᵀ], an
# InverseWishart with p - n - 1 degrees of freedom, improper when n ≥ p.

@define_message_update_rule(
    node = MatrixNormal, target = :U,
    args = (m[:out]::PointMass, m[:M]::PointMass, m[:V]::PointMass),
    body = (args) -> ((n, p) = size(mean(args.m[:out])); InverseWishartFast(p - n - 1, row_scatter(args.m[:out], args.m[:M], cholinv(mean(args.m[:V]))))),
)

@define_message_update_rule(
    node = MatrixNormal, target = :U,
    args = (q[:out]::PointOrMatrixNormal, q[:M]::PointOrMatrixNormal, q[:V]::Any),
    body = (args) -> ((n, p) = size(mean(args.q[:out])); InverseWishartFast(p - n - 1, row_scatter(args.q[:out], args.q[:M], mean(cholinv, args.q[:V])))),
)
