@define_marginal_update_rule(
    node = NormalMeanPrecision, target = (:out, :μ),
    args = (m[:out]::UnivariateNormalDistributionsFamily, m[:μ]::UnivariateNormalDistributionsFamily, q[:τ]::Any),
    body = (args) -> begin
        xi_out, W_out = weightedmean_precision(args.m[:out])
        xi_μ, W_μ = weightedmean_precision(args.m[:μ])
        W_bar = mean(args.q[:τ])
        MvNormalWeightedMeanPrecision([xi_out; xi_μ], [W_out + W_bar -W_bar; -W_bar W_μ + W_bar])
    end,
)

# A point-mass message splits the cluster: each block is its own marginal. A precision's
# expectation is the right one to use.

@define_marginal_update_rule(
    node = NormalMeanPrecision, target = (:out, :μ),
    args = (m[:out]::PointMass, m[:μ]::UnivariateNormalDistributionsFamily, q[:τ]::Any),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => args.m[:out],
            (:μ,) => prod(ClosedProd(), NormalMeanPrecision(mean(args.m[:out]), mean(args.q[:τ])), args.m[:μ]),
        ),
        args.m[:out], args.m[:μ], args.q[:τ],
    ),
)

@define_marginal_update_rule(
    node = NormalMeanPrecision, target = (:out, :μ),
    args = (m[:out]::UnivariateNormalDistributionsFamily, m[:μ]::PointMass, q[:τ]::Any),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => prod(ClosedProd(), NormalMeanPrecision(mean(args.m[:μ]), mean(args.q[:τ])), args.m[:out]),
            (:μ,) => args.m[:μ],
        ),
        args.m[:out], args.m[:μ], args.q[:τ],
    ),
)

# The whole node as one cluster, under belief propagation.

@define_marginal_update_rule(
    node = NormalMeanPrecision, target = (:out, :μ, :τ),
    args = (m[:out]::UnivariateNormalDistributionsFamily, m[:μ]::PointMass, m[:τ]::PointMass),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => prod(ClosedProd(), NormalMeanPrecision(mean(args.m[:μ]), mean(args.m[:τ])), args.m[:out]),
            (:μ,) => args.m[:μ],
            (:τ,) => args.m[:τ],
        ),
        args.m[:out], args.m[:μ], args.m[:τ],
    ),
)

@define_marginal_update_rule(
    node = NormalMeanPrecision, target = (:out, :μ, :τ),
    args = (m[:out]::PointMass, m[:μ]::UnivariateNormalDistributionsFamily, m[:τ]::PointMass),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => args.m[:out],
            (:μ,) => prod(ClosedProd(), args.m[:μ], NormalMeanPrecision(mean(args.m[:out]), mean(args.m[:τ]))),
            (:τ,) => args.m[:τ],
        ),
        args.m[:out], args.m[:μ], args.m[:τ],
    ),
)

@define_marginal_update_rule(
    node = NormalMeanPrecision, target = (:out, :μ, :τ),
    args = (m[:out]::UnivariateNormalDistributionsFamily, m[:μ]::UnivariateNormalDistributionsFamily, m[:τ]::PointMass),
    body = (args) -> begin
        xi_out, W_out = weightedmean_precision(args.m[:out])
        xi_μ, W_μ = weightedmean_precision(args.m[:μ])
        W_bar = mean(args.m[:τ])
        joint = MvNormalWeightedMeanPrecision([xi_out; xi_μ], [W_out + W_bar -W_bar; -W_bar W_μ + W_bar])
        promoted_cluster(FactorizedCluster((:out, :μ) => joint, (:τ,) => args.m[:τ]), args.m[:out], args.m[:μ], args.m[:τ])
    end,
)
