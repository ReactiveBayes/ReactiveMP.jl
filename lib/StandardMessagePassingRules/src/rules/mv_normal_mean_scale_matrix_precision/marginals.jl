# As MvNormalMeanPrecision's, with the precision E[γ]·E[G] coupling `out` and `μ`.

@define_marginal_update_rule(
    node = MvNormalMeanScaleMatrixPrecision, target = (:out, :μ),
    args = (m[:out]::MultivariateNormalDistributionsFamily, m[:μ]::MultivariateNormalDistributionsFamily, q[:γ]::Any, q[:G]::Any),
    body = (args) -> mv_coupled(args.m[:out], args.m[:μ], mean(args.q[:γ]) * mean(args.q[:G])),
)

@define_marginal_update_rule(
    node = MvNormalMeanScaleMatrixPrecision, target = (:out, :μ),
    args = (m[:out]::PointMass, m[:μ]::MultivariateNormalDistributionsFamily, q[:γ]::Any, q[:G]::Any),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => args.m[:out],
            (:μ,) => prod(
                ClosedProd(), MvNormalMeanPrecision(mean(args.m[:out]), mean(args.q[:γ]) * mean(args.q[:G])), args.m[:μ],
            ),
        ),
        args.m[:out], args.m[:μ], args.q[:γ], args.q[:G],
    ),
)

@define_marginal_update_rule(
    node = MvNormalMeanScaleMatrixPrecision, target = (:out, :μ),
    args = (m[:out]::MultivariateNormalDistributionsFamily, m[:μ]::PointMass, q[:γ]::Any, q[:G]::Any),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => prod(
                ClosedProd(), MvNormalMeanPrecision(mean(args.m[:μ]), mean(args.q[:γ]) * mean(args.q[:G])), args.m[:out],
            ),
            (:μ,) => args.m[:μ],
        ),
        args.m[:out], args.m[:μ], args.q[:γ], args.q[:G],
    ),
)

@define_marginal_update_rule(
    node = MvNormalMeanScaleMatrixPrecision, target = (:out, :μ, :γ, :G),
    args = (m[:out]::MultivariateNormalDistributionsFamily, m[:μ]::PointMass, m[:γ]::PointMass, m[:G]::PointMass),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => prod(
                ClosedProd(), MvNormalMeanPrecision(mean(args.m[:μ]), mean(args.m[:γ]) * mean(args.m[:G])), args.m[:out],
            ),
            (:μ,) => args.m[:μ],
            (:γ,) => args.m[:γ],
            (:G,) => args.m[:G],
        ),
        args.m[:out], args.m[:μ], args.m[:γ], args.m[:G],
    ),
)

@define_marginal_update_rule(
    node = MvNormalMeanScaleMatrixPrecision, target = (:out, :μ, :γ, :G),
    args = (m[:out]::PointMass, m[:μ]::MultivariateNormalDistributionsFamily, m[:γ]::PointMass, m[:G]::PointMass),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => args.m[:out],
            (:μ,) => prod(
                ClosedProd(), args.m[:μ], MvNormalMeanPrecision(mean(args.m[:out]), mean(args.m[:γ]) * mean(args.m[:G])),
            ),
            (:γ,) => args.m[:γ],
            (:G,) => args.m[:G],
        ),
        args.m[:out], args.m[:μ], args.m[:γ], args.m[:G],
    ),
)

@define_marginal_update_rule(
    node = MvNormalMeanScaleMatrixPrecision, target = (:out, :μ, :γ, :G),
    args = (m[:out]::MultivariateNormalDistributionsFamily, m[:μ]::MultivariateNormalDistributionsFamily, m[:γ]::PointMass, m[:G]::PointMass),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out, :μ) => mv_coupled(args.m[:out], args.m[:μ], mean(args.m[:γ]) * mean(args.m[:G])),
            (:γ,) => args.m[:γ],
            (:G,) => args.m[:G],
        ),
        args.m[:out], args.m[:μ], args.m[:γ], args.m[:G],
    ),
)
