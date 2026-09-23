# The whole node with every input known.
@define_marginal_update_rule(
    node = MatrixNormal, target = (:out, :M, :U, :V),
    args = (m[:out]::PointMass, m[:M]::PointMass, m[:U]::PointMass, m[:V]::PointMass),
    body = (args) -> promoted_cluster(
        FactorizedCluster((:out,) => args.m[:out], (:M,) => args.m[:M], (:U,) => args.m[:U], (:V,) => args.m[:V]),
        args.m[:out], args.m[:M], args.m[:U], args.m[:V],
    ),
)
