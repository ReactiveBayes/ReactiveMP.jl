# Belief propagation.

@define_message_update_rule(
    node = MvNormalMeanPrecision, target = :out,
    args = (m[:μ]::PointMass, m[:Λ]::PointMass),
    logscale = 0,
    body = (args) -> MvNormalMeanPrecision(mean(args.m[:μ]), mean(args.m[:Λ])),
)

@define_message_update_rule(
    node = MvNormalMeanPrecision, target = :out,
    args = (m[:μ]::MultivariateNormalDistributionsFamily, m[:Λ]::PointMass),
    logscale = 0,
    body = (args) -> begin
        μ, V = mean_cov(args.m[:μ])
        MvNormalMeanCovariance(μ, V + cholinv(mean(args.m[:Λ])))
    end,
)

# Variational: a `q_Λ` contributes the precision E[Λ].

@define_message_update_rule(
    node = MvNormalMeanPrecision, target = :out,
    args = (q[:μ]::PointMass, q[:Λ]::PointMass),
    logscale = 0,
    body = (args) -> MvNormalMeanPrecision(mean(args.q[:μ]), mean(args.q[:Λ])),
)

@define_message_update_rule(
    node = MvNormalMeanPrecision, target = :out,
    args = (q[:μ]::Any, q[:Λ]::Any),
    body = (args) -> MvNormalMeanPrecision(mean(args.q[:μ]), mean(args.q[:Λ])),
)

@define_message_update_rule(
    node = MvNormalMeanPrecision, target = :out,
    args = (m[:μ]::PointMass, q[:Λ]::Any),
    body = (args) -> MvNormalMeanPrecision(mean(args.m[:μ]), mean(args.q[:Λ])),
)

@define_message_update_rule(
    node = MvNormalMeanPrecision, target = :out,
    args = (m[:μ]::MultivariateNormalDistributionsFamily, q[:Λ]::Any),
    body = (args) -> begin
        μ, V = mean_cov(args.m[:μ])
        MvNormalMeanCovariance(μ, V + cholinv(mean(args.q[:Λ])))
    end,
)

# A Wishart q_Λ = W(df, S) has E[Λ]⁻¹ = S⁻¹/df, from the factor S already holds.
@define_message_update_rule(
    node = MvNormalMeanPrecision, target = :out,
    args = (m[:μ]::MultivariateNormalDistributionsFamily, q[:Λ]::Wishart),
    body = (args) -> begin
        μ, V = mean_cov(args.m[:μ])
        MvNormalMeanCovariance(μ, V + inv(args.q[:Λ].S.chol) ./ args.q[:Λ].df)
    end,
)
