@define_marginal_update_rule(
    node = NormalMeanVariance, target = (:out, :μ),
    args = (m[:out]::UnivariateNormalDistributionsFamily, m[:μ]::UnivariateNormalDistributionsFamily, q[:v]::Any),
    body = (args) -> begin
        xi_out, W_out = weightedmean_precision(args.m[:out])
        xi_μ, W_μ = weightedmean_precision(args.m[:μ])
        W_bar = mean(inv, args.q[:v])
        MvNormalWeightedMeanPrecision([xi_out; xi_μ], [W_out + W_bar -W_bar; -W_bar W_μ + W_bar])
    end,
)

# A point-mass message splits the cluster: each block is its own marginal. As in the rules
# above, a `q_v` contributes the variance 1/E[1/v].

@define_marginal_update_rule(
    node = NormalMeanVariance, target = (:out, :μ),
    args = (m[:out]::PointMass, m[:μ]::UnivariateNormalDistributionsFamily, q[:v]::Any),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => args.m[:out],
            (:μ,) => prod(ClosedProd(), NormalMeanVariance(mean(args.m[:out]), variational_variance(args.q[:v])), args.m[:μ]),
        ),
        args.m[:out], args.m[:μ], args.q[:v],
    ),
)

@define_marginal_update_rule(
    node = NormalMeanVariance, target = (:out, :μ),
    args = (m[:out]::UnivariateNormalDistributionsFamily, m[:μ]::PointMass, q[:v]::Any),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => prod(ClosedProd(), NormalMeanVariance(mean(args.m[:μ]), variational_variance(args.q[:v])), args.m[:out]),
            (:μ,) => args.m[:μ],
        ),
        args.m[:out], args.m[:μ], args.q[:v],
    ),
)

# The whole node as one cluster, under belief propagation.

@define_marginal_update_rule(
    node = NormalMeanVariance, target = (:out, :μ, :v),
    args = (m[:out]::NormalDistributionsFamily, m[:μ]::PointMass, m[:v]::PointMass),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => prod(ClosedProd(), NormalMeanVariance(mean(args.m[:μ]), mean(args.m[:v])), args.m[:out]),
            (:μ,) => args.m[:μ],
            (:v,) => args.m[:v],
        ),
        args.m[:out], args.m[:μ], args.m[:v],
    ),
)

@define_marginal_update_rule(
    node = NormalMeanVariance, target = (:out, :μ, :v),
    args = (m[:out]::PointMass, m[:μ]::NormalDistributionsFamily, m[:v]::PointMass),
    body = (args) -> promoted_cluster(
        FactorizedCluster(
            (:out,) => args.m[:out],
            (:μ,) => prod(ClosedProd(), args.m[:μ], NormalMeanVariance(mean(args.m[:out]), mean(args.m[:v]))),
            (:v,) => args.m[:v],
        ),
        args.m[:out], args.m[:μ], args.m[:v],
    ),
)

@define_marginal_update_rule(
    node = NormalMeanVariance, target = (:out, :μ, :v),
    args = (m[:out]::UnivariateNormalDistributionsFamily, m[:μ]::UnivariateNormalDistributionsFamily, m[:v]::PointMass),
    body = (args) -> begin
        xi_out, W_out = weightedmean_precision(args.m[:out])
        xi_μ, W_μ = weightedmean_precision(args.m[:μ])
        W_bar = inv(mean(args.m[:v]))
        joint = MvNormalWeightedMeanPrecision([xi_out; xi_μ], [W_out + W_bar -W_bar; -W_bar W_μ + W_bar])
        promoted_cluster(FactorizedCluster((:out, :μ) => joint, (:v,) => args.m[:v]), args.m[:out], args.m[:μ], args.m[:v])
    end,
)
