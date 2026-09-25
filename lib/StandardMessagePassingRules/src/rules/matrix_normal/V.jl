# The likelihood of V: |V|^(-n/2) exp(-tr(V⁻¹ Ψ)/2) with Ψ = E[(out - M)ᵀ E[U⁻¹] (out - M)], an
# InverseWishart with n - p - 1 degrees of freedom, improper when p ≥ n.

@define_message_update_rule(
    node = MatrixNormal, target = :V,
    args = (m[:out]::PointMass, m[:M]::PointMass, m[:U]::PointMass),
    body = (args) -> ((n, p) = size(mean(args.m[:out])); InverseWishartFast(n - p - 1, column_scatter(args.m[:out], args.m[:M], cholinv(mean(args.m[:U]))))),
)

@define_message_update_rule(
    node = MatrixNormal, target = :V,
    args = (q[:out]::PointOrMatrixNormal, q[:M]::PointOrMatrixNormal, q[:U]::Any),
    body = (args) -> ((n, p) = size(mean(args.q[:out])); InverseWishartFast(n - p - 1, column_scatter(args.q[:out], args.q[:M], mean(cholinv, args.q[:U])))),
)
