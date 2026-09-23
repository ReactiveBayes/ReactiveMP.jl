@define_marginal_update_rule(
    node = Beta, target = (:out, :a, :b),
    args = (m[:out]::Beta, m[:a]::PointMass, m[:b]::PointMass),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => prod(ClosedProd(), Beta(mean(args.m[:a]), mean(args.m[:b])), args.m[:out]),
            (:a,) => args.m[:a],
            (:b,) => args.m[:b],
        ),
        args.m[:out], args.m[:a], args.m[:b],
    ),
)
