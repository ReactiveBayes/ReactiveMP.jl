@define_message_update_rule(
    node = MvNormalWeightedMeanPrecision, target = :out,
    args = (m[:ξ]::PointMass, m[:Λ]::PointMass),
    logscale = 0,
    body = (args) -> MvNormalWeightedMeanPrecision(mean(args.m[:ξ]), mean(args.m[:Λ])),
)

@define_message_update_rule(
    node = MvNormalWeightedMeanPrecision, target = :out,
    args = (q[:ξ]::Any, q[:Λ]::Any),
    body = (args) -> MvNormalWeightedMeanPrecision(mean(args.q[:ξ]), mean(args.q[:Λ])),
)
