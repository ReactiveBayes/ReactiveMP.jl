# NormalMeanVariance and NormalMeanPrecision with an ExponentialLinearQuadratic message on `out`,
# as a GCV node's `y` sends one: the message is reduced to a normal of its moments and the
# normal's own rule applied, its formulas written out here with Standard's helpers, since a rule
# does not call another. v6 wrote these as `@call_rule`s and never tested them.

moments_normal(m::ExponentialLinearQuadratic) = NormalMeanVariance(mean_var(m)...)

@define_message_update_rule(
    node = NormalMeanVariance, target = :μ, args = (m[:out]::ExponentialLinearQuadratic, m[:v]::PointMass),
    body = (args) -> ((m, v) = mean_var(args.m[:out]); NormalMeanVariance(m, v + mean(args.m[:v]))),
)

@define_message_update_rule(
    node = NormalMeanVariance, target = :μ, args = (m[:out]::ExponentialLinearQuadratic, q[:v]::Any),
    body = (args) -> ((m, v) = mean_var(args.m[:out]); NormalMeanVariance(m, v + StandardMessagePassingRules.variational_variance(args.q[:v]))),
)

@define_message_update_rule(
    node = NormalMeanPrecision, target = :μ, args = (m[:out]::ExponentialLinearQuadratic, m[:τ]::PointMass),
    body = (args) -> ((m, v) = mean_var(args.m[:out]); NormalMeanVariance(m, v + inv(mean(args.m[:τ])))),
)

@define_message_update_rule(
    node = NormalMeanPrecision, target = :μ, args = (m[:out]::ExponentialLinearQuadratic, q[:τ]::Any),
    body = (args) -> ((m, v) = mean_var(args.m[:out]); NormalMeanVariance(m, v + inv(mean(args.q[:τ])))),
)

# The joint of `out` and `μ`: the two messages coupled by the precision W̄.
function coupled_joint(m_out, m_μ, W_bar)
    xi_out, W_out = weightedmean_precision(moments_normal(m_out))
    xi_μ, W_μ = weightedmean_precision(m_μ)
    return MvNormalWeightedMeanPrecision([xi_out; xi_μ], StandardMessagePassingRules.coupled_precision(W_out, W_μ, W_bar))
end

@define_marginal_update_rule(
    node = NormalMeanVariance, target = (:out, :μ, :v),
    args = (m[:out]::ExponentialLinearQuadratic, m[:μ]::UnivariateNormalDistributionsFamily, m[:v]::PointMass),
    body = (args) -> StandardMessagePassingRules.promoted_cluster(
        FactorizedCluster((:out, :μ) => coupled_joint(args.m[:out], args.m[:μ], inv(mean(args.m[:v]))), (:v,) => args.m[:v]),
        args.m[:μ], args.m[:v],
    ),
)

@define_marginal_update_rule(
    node = NormalMeanVariance, target = (:out, :μ),
    args = (m[:out]::ExponentialLinearQuadratic, m[:μ]::UnivariateNormalDistributionsFamily, q[:v]::Any),
    body = (args) -> coupled_joint(args.m[:out], args.m[:μ], mean(inv, args.q[:v])),
)

@define_marginal_update_rule(
    node = NormalMeanPrecision, target = (:out, :μ, :τ),
    args = (m[:out]::ExponentialLinearQuadratic, m[:μ]::UnivariateNormalDistributionsFamily, m[:τ]::PointMass),
    body = (args) -> StandardMessagePassingRules.promoted_cluster(
        FactorizedCluster((:out, :μ) => coupled_joint(args.m[:out], args.m[:μ], mean(args.m[:τ])), (:τ,) => args.m[:τ]),
        args.m[:μ], args.m[:τ],
    ),
)

@define_marginal_update_rule(
    node = NormalMeanPrecision, target = (:out, :μ),
    args = (m[:out]::ExponentialLinearQuadratic, m[:μ]::UnivariateNormalDistributionsFamily, q[:τ]::Any),
    body = (args) -> coupled_joint(args.m[:out], args.m[:μ], mean(args.q[:τ])),
)
