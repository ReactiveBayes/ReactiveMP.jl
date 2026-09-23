@define_marginal_update_rule(
    node = GammaShapeRate, target = (:out, :α, :β),
    args = (m[:out]::GammaDistributionsFamily, m[:α]::PointMass, m[:β]::PointMass),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => prod(ClosedProd(), GammaShapeRate(mean(args.m[:α]), mean(args.m[:β])), args.m[:out]),
            (:α,) => args.m[:α],
            (:β,) => args.m[:β],
        ),
        args.m[:out], args.m[:α], args.m[:β],
    ),
)
