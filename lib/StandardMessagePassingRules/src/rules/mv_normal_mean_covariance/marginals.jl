# The joint of `out` and `μ` from their messages, and a split cluster when one of them is a
# point mass. A `q_Σ` contributes the precision E[Σ⁻¹].

mv_coupled(m_out, m_μ, W_bar) = begin
    xi_out, W_out = weightedmean_precision(m_out)
    xi_μ, W_μ = weightedmean_precision(m_μ)
    MvNormalWeightedMeanPrecision([xi_out; xi_μ], coupled_precision(W_out, W_μ, W_bar))
end

@define_marginal_update_rule(
    node = MvNormalMeanCovariance, target = (:out, :μ),
    args = (m[:out]::MultivariateNormalDistributionsFamily, m[:μ]::MultivariateNormalDistributionsFamily, q[:Σ]::Any),
    body = (args) -> mv_coupled(args.m[:out], args.m[:μ], mean(cholinv, args.q[:Σ])),
)

@define_marginal_update_rule(
    node = MvNormalMeanCovariance, target = (:out, :μ),
    args = (m[:out]::PointMass, m[:μ]::MultivariateNormalDistributionsFamily, q[:Σ]::Any),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => args.m[:out],
            (:μ,) => prod(ClosedProd(), MvNormalMeanCovariance(mean(args.m[:out]), variational_covariance(args.q[:Σ])), args.m[:μ]),
        ),
        args.m[:out], args.m[:μ], args.q[:Σ],
    ),
)

@define_marginal_update_rule(
    node = MvNormalMeanCovariance, target = (:out, :μ),
    args = (m[:out]::MultivariateNormalDistributionsFamily, m[:μ]::PointMass, q[:Σ]::Any),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => prod(ClosedProd(), MvNormalMeanCovariance(mean(args.m[:μ]), variational_covariance(args.q[:Σ])), args.m[:out]),
            (:μ,) => args.m[:μ],
        ),
        args.m[:out], args.m[:μ], args.q[:Σ],
    ),
)

@define_marginal_update_rule(
    node = MvNormalMeanCovariance, target = (:out, :μ, :Σ),
    args = (m[:out]::MultivariateNormalDistributionsFamily, m[:μ]::PointMass, m[:Σ]::PointMass),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => prod(ClosedProd(), MvNormalMeanCovariance(mean(args.m[:μ]), mean(args.m[:Σ])), args.m[:out]),
            (:μ,) => args.m[:μ],
            (:Σ,) => args.m[:Σ],
        ),
        args.m[:out], args.m[:μ], args.m[:Σ],
    ),
)

@define_marginal_update_rule(
    node = MvNormalMeanCovariance, target = (:out, :μ, :Σ),
    args = (m[:out]::PointMass, m[:μ]::MultivariateNormalDistributionsFamily, m[:Σ]::PointMass),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => args.m[:out],
            (:μ,) => prod(ClosedProd(), args.m[:μ], MvNormalMeanCovariance(mean(args.m[:out]), mean(args.m[:Σ]))),
            (:Σ,) => args.m[:Σ],
        ),
        args.m[:out], args.m[:μ], args.m[:Σ],
    ),
)

@define_marginal_update_rule(
    node = MvNormalMeanCovariance, target = (:out, :μ, :Σ),
    args = (m[:out]::MultivariateNormalDistributionsFamily, m[:μ]::MultivariateNormalDistributionsFamily, m[:Σ]::PointMass),
    body = (args) -> promoted_cluster(
        FactorizedCluster((:out, :μ) => mv_coupled(args.m[:out], args.m[:μ], cholinv(mean(args.m[:Σ]))), (:Σ,) => args.m[:Σ]),
        args.m[:out], args.m[:μ], args.m[:Σ],
    ),
)
