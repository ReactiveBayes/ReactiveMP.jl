@define_marginal_update_rule(
    node = GammaInverse, target = (:out, :α, :θ),
    args = (m[:out]::GammaInverse, m[:α]::PointMass, m[:θ]::PointMass),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => prod(ClosedProd(), GammaInverse(mean(args.m[:α]), mean(args.m[:θ])), args.m[:out]),
            (:α,) => args.m[:α],
            (:θ,) => args.m[:θ],
        ),
        args.m[:out], args.m[:α], args.m[:θ],
    ),
)
