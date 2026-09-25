# Belief propagation.

@define_message_update_rule(
    node = MvNormalMeanCovariance, target = :out,
    args = (m[:μ]::PointMass, m[:Σ]::PointMass),
    body = (args) -> MvNormalMeanCovariance(mean(args.m[:μ]), mean(args.m[:Σ])),
)

@define_message_update_rule(
    node = MvNormalMeanCovariance, target = :out,
    args = (m[:μ]::MultivariateNormalDistributionsFamily, m[:Σ]::PointMass),
    body = (args, ann) -> begin
        annotate!(ann, :logscale, 0)
        μ, V = mean_cov(args.m[:μ])
        MvNormalMeanCovariance(μ, V + mean(args.m[:Σ]))
    end,
)

# Variational; a `q_Σ` contributes E[Σ⁻¹]⁻¹.

@define_message_update_rule(
    node = MvNormalMeanCovariance, target = :out,
    args = (q[:μ]::PointMass, q[:Σ]::PointMass),
    body = (args) -> MvNormalMeanCovariance(mean(args.q[:μ]), mean(args.q[:Σ])),
)

@define_message_update_rule(
    node = MvNormalMeanCovariance, target = :out,
    args = (q[:μ]::Any, q[:Σ]::Any),
    body = (args) -> MvNormalMeanCovariance(mean(args.q[:μ]), variational_covariance(args.q[:Σ])),
)

@define_message_update_rule(
    node = MvNormalMeanCovariance, target = :out,
    args = (m[:μ]::PointMass, q[:Σ]::Any),
    body = (args) -> MvNormalMeanCovariance(mean(args.m[:μ]), variational_covariance(args.q[:Σ])),
)

@define_message_update_rule(
    node = MvNormalMeanCovariance, target = :out,
    args = (m[:μ]::MultivariateNormalDistributionsFamily, q[:Σ]::Any),
    body = (args) -> begin
        μ, V = mean_cov(args.m[:μ])
        MvNormalMeanCovariance(μ, V + variational_covariance(args.q[:Σ]))
    end,
)
