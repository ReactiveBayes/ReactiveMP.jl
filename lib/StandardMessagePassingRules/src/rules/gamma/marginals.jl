@define_marginal_update_rule(
    node = Gamma, target = (:out, :α, :θ),
    args = (m[:out]::GammaDistributionsFamily, m[:α]::PointMass, m[:θ]::PointMass),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => prod(ClosedProd(), Gamma(mean(args.m[:α]), mean(args.m[:θ])), args.m[:out]),
            (:α,) => args.m[:α],
            (:θ,) => args.m[:θ],
        ),
        args.m[:out], args.m[:α], args.m[:θ],
    ),
)
