# As MvNormalMeanCovariance's, with the precision E[Λ] coupling `out` and `μ`.

@define_marginal_update_rule(
    node = MvNormalMeanPrecision, target = (:out, :μ),
    args = (m[:out]::MultivariateNormalDistributionsFamily, m[:μ]::MultivariateNormalDistributionsFamily, q[:Λ]::Any),
    body = (args) -> mv_coupled(args.m[:out], args.m[:μ], mean(args.q[:Λ])),
)

@define_marginal_update_rule(
    node = MvNormalMeanPrecision, target = (:out, :μ),
    args = (m[:out]::PointMass, m[:μ]::MultivariateNormalDistributionsFamily, q[:Λ]::Any),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => args.m[:out],
            (:μ,) => prod(ClosedProd(), MvNormalMeanPrecision(mean(args.m[:out]), mean(args.q[:Λ])), args.m[:μ]),
        ),
        args.m[:out], args.m[:μ], args.q[:Λ],
    ),
)

@define_marginal_update_rule(
    node = MvNormalMeanPrecision, target = (:out, :μ),
    args = (m[:out]::MultivariateNormalDistributionsFamily, m[:μ]::PointMass, q[:Λ]::Any),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => prod(ClosedProd(), MvNormalMeanPrecision(mean(args.m[:μ]), mean(args.q[:Λ])), args.m[:out]),
            (:μ,) => args.m[:μ],
        ),
        args.m[:out], args.m[:μ], args.q[:Λ],
    ),
)

@define_marginal_update_rule(
    node = MvNormalMeanPrecision, target = (:out, :μ, :Λ),
    args = (m[:out]::MultivariateNormalDistributionsFamily, m[:μ]::PointMass, m[:Λ]::PointMass),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => prod(ClosedProd(), MvNormalMeanPrecision(mean(args.m[:μ]), mean(args.m[:Λ])), args.m[:out]),
            (:μ,) => args.m[:μ],
            (:Λ,) => args.m[:Λ],
        ),
        args.m[:out], args.m[:μ], args.m[:Λ],
    ),
)

@define_marginal_update_rule(
    node = MvNormalMeanPrecision, target = (:out, :μ, :Λ),
    args = (m[:out]::PointMass, m[:μ]::MultivariateNormalDistributionsFamily, m[:Λ]::PointMass),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => args.m[:out],
            (:μ,) => prod(ClosedProd(), args.m[:μ], MvNormalMeanPrecision(mean(args.m[:out]), mean(args.m[:Λ]))),
            (:Λ,) => args.m[:Λ],
        ),
        args.m[:out], args.m[:μ], args.m[:Λ],
    ),
)

@define_marginal_update_rule(
    node = MvNormalMeanPrecision, target = (:out, :μ, :Λ),
    args = (m[:out]::MultivariateNormalDistributionsFamily, m[:μ]::MultivariateNormalDistributionsFamily, m[:Λ]::PointMass),
    body = (args) -> promoted_cluster(
        FactorizedCluster((:out, :μ) => mv_coupled(args.m[:out], args.m[:μ], mean(args.m[:Λ])), (:Λ,) => args.m[:Λ]),
        args.m[:out], args.m[:μ], args.m[:Λ],
    ),
)
