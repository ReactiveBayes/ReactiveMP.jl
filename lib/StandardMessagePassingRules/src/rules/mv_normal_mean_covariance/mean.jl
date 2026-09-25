# Belief propagation.

@define_message_update_rule(
    node = MvNormalMeanCovariance, target = :μ,
    args = (m[:out]::PointMass, m[:Σ]::PointMass),
    body = (args) -> MvNormalMeanCovariance(mean(args.m[:out]), mean(args.m[:Σ])),
)

@define_message_update_rule(
    node = MvNormalMeanCovariance, target = :μ,
    args = (m[:out]::MultivariateNormalDistributionsFamily, m[:Σ]::PointMass),
    body = (args, ann) -> begin
        annotate!(ann, :logscale, 0)
        μ, V = mean_cov(args.m[:out])
        MvNormalMeanCovariance(μ, V + mean(args.m[:Σ]))
    end,
)

# Variational; a `q_Σ` contributes E[Σ⁻¹]⁻¹.

@define_message_update_rule(
    node = MvNormalMeanCovariance, target = :μ,
    args = (q[:out]::PointMass, q[:Σ]::PointMass),
    body = (args) -> MvNormalMeanCovariance(mean(args.q[:out]), mean(args.q[:Σ])),
)

@define_message_update_rule(
    node = MvNormalMeanCovariance, target = :μ,
    args = (q[:out]::Any, q[:Σ]::Any),
    body = (args) -> MvNormalMeanCovariance(mean(args.q[:out]), variational_covariance(args.q[:Σ])),
)

@define_message_update_rule(
    node = MvNormalMeanCovariance, target = :μ,
    args = (m[:out]::PointMass, q[:Σ]::Any),
    body = (args) -> MvNormalMeanCovariance(mean(args.m[:out]), variational_covariance(args.q[:Σ])),
)

@define_message_update_rule(
    node = MvNormalMeanCovariance, target = :μ,
    args = (m[:out]::MultivariateNormalDistributionsFamily, q[:Σ]::Any),
    body = (args) -> begin
        μ, V = mean_cov(args.m[:out])
        MvNormalMeanCovariance(μ, V + variational_covariance(args.q[:Σ]))
    end,
)
