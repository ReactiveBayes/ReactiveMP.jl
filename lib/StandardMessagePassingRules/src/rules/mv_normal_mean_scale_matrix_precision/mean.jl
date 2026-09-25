# Variational only: `q_γ` and `q_G` contribute the precision E[γ]·E[G].

@define_message_update_rule(
    node = MvNormalMeanScaleMatrixPrecision, target = :μ,
    args = (q[:out]::Any, q[:γ]::Any, q[:G]::Any),
    body = (args) -> MvNormalMeanPrecision(mean(args.q[:out]), mean(args.q[:γ]) * mean(args.q[:G])),
)

@define_message_update_rule(
    node = MvNormalMeanScaleMatrixPrecision, target = :μ,
    args = (m[:out]::MultivariateNormalDistributionsFamily, q[:γ]::Any, q[:G]::Any),
    body = (args) -> series_precision(args.m[:out], mean(args.q[:γ]) * mean(args.q[:G])),
)
