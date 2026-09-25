# Belief propagation.

@define_message_update_rule(
    node = MvNormalMeanPrecision, target = :μ,
    args = (m[:out]::PointMass, m[:Λ]::PointMass),
    logscale = 0,
    body = (args) -> MvNormalMeanPrecision(mean(args.m[:out]), mean(args.m[:Λ])),
)

@define_message_update_rule(
    node = MvNormalMeanPrecision, target = :μ,
    args = (m[:out]::MultivariateNormalDistributionsFamily, m[:Λ]::PointMass),
    logscale = 0,
    body = (args) -> begin
        μ, V = mean_cov(args.m[:out])
        MvNormalMeanCovariance(μ, V + cholinv(mean(args.m[:Λ])))
    end,
)

# Variational: a `q_Λ` contributes the precision E[Λ].

@define_message_update_rule(
    node = MvNormalMeanPrecision, target = :μ,
    args = (q[:out]::PointMass, q[:Λ]::PointMass),
    logscale = 0,
    body = (args) -> MvNormalMeanPrecision(mean(args.q[:out]), mean(args.q[:Λ])),
)

@define_message_update_rule(
    node = MvNormalMeanPrecision, target = :μ,
    args = (q[:out]::Any, q[:Λ]::Any),
    body = (args) -> MvNormalMeanPrecision(mean(args.q[:out]), mean(args.q[:Λ])),
)

@define_message_update_rule(
    node = MvNormalMeanPrecision, target = :μ,
    args = (m[:out]::PointMass, q[:Λ]::Any),
    body = (args) -> MvNormalMeanPrecision(mean(args.m[:out]), mean(args.q[:Λ])),
)

@define_message_update_rule(
    node = MvNormalMeanPrecision, target = :μ,
    args = (m[:out]::MultivariateNormalDistributionsFamily, q[:Λ]::Any),
    body = (args) -> begin
        μ, V = mean_cov(args.m[:out])
        MvNormalMeanCovariance(μ, V + cholinv(mean(args.q[:Λ])))
    end,
)

# A Wishart q_Λ = W(df, S) has E[Λ]⁻¹ = S⁻¹/df, from the factor S already holds.
@define_message_update_rule(
    node = MvNormalMeanPrecision, target = :μ,
    args = (m[:out]::MultivariateNormalDistributionsFamily, q[:Λ]::Wishart),
    body = (args) -> begin
        μ, V = mean_cov(args.m[:out])
        MvNormalMeanCovariance(μ, V + inv(args.q[:Λ].S.chol) ./ args.q[:Λ].df)
    end,
)
