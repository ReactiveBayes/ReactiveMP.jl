# Variational only: a `q_γ` contributes the precision E[γ]·I.

@define_message_update_rule(
    node = MvNormalMeanScalePrecision, target = :out,
    args = (q[:μ]::Any, q[:γ]::Any),
    body = (args) -> MvNormalMeanScalePrecision(mean(args.q[:μ]), mean(args.q[:γ])),
)

@define_message_update_rule(
    node = MvNormalMeanScalePrecision, target = :out,
    args = (m[:μ]::MultivariateNormalDistributionsFamily, q[:γ]::Any),
    body = (args) -> begin
        μ, V = mean_cov(args.m[:μ])
        MvNormalMeanCovariance(μ, V + inv(mean(args.q[:γ])) * diageye(eltype(μ), length(μ)))
    end,
)

# A scale-precision message stays one: precisions l and r in series give lr/(l + r).
@define_message_update_rule(
    node = MvNormalMeanScalePrecision, target = :out,
    args = (m[:μ]::MvNormalMeanScalePrecision, q[:γ]::Any),
    body = (args) -> begin
        l, r = args.m[:μ].γ, mean(args.q[:γ])
        MvNormalMeanScalePrecision(mean(args.m[:μ]), (l * r) / (l + r))
    end,
)
