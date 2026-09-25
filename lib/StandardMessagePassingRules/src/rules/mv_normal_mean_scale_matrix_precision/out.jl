# Variational only: `q_γ` and `q_G` contribute the precision E[γ]·E[G].

@define_message_update_rule(
    node = MvNormalMeanScaleMatrixPrecision, target = :out,
    args = (q[:μ]::Any, q[:γ]::Any, q[:G]::Any),
    body = (args) -> MvNormalMeanPrecision(mean(args.q[:μ]), mean(args.q[:γ]) * mean(args.q[:G])),
)

@define_message_update_rule(
    node = MvNormalMeanScaleMatrixPrecision, target = :out,
    args = (m[:μ]::MultivariateNormalDistributionsFamily, q[:γ]::Any, q[:G]::Any),
    body = (args) -> series_precision(args.m[:μ], mean(args.q[:γ]) * mean(args.q[:G])),
)
