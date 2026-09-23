@define_marginal_update_rule(
    node = Poisson, target = (:out, :l),
    args = (m[:out]::PointMass, m[:l]::Gamma),
    body = (args) -> promoted_cluster(
        FactorizedCluster((:out,) => args.m[:out], (:l,) => prod(ClosedProd(), Gamma(mean(args.m[:out]) + 1, 1), args.m[:l])),
        args.m[:out], args.m[:l],
    ),
)
