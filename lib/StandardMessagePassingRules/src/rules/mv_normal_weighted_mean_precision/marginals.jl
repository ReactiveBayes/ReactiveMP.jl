# The cluster's blocks are labelled `out`, `ξ` and `Λ`, by interface.
@define_marginal_update_rule(
    node = MvNormalWeightedMeanPrecision, target = (:out, :ξ, :Λ),
    args = (m[:out]::MultivariateNormalDistributionsFamily, m[:ξ]::PointMass, m[:Λ]::PointMass),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => prod(ClosedProd(), MvNormalWeightedMeanPrecision(mean(args.m[:ξ]), mean(args.m[:Λ])), args.m[:out]),
            (:ξ,) => args.m[:ξ],
            (:Λ,) => args.m[:Λ],
        ),
        args.m[:out], args.m[:ξ], args.m[:Λ],
    ),
)
