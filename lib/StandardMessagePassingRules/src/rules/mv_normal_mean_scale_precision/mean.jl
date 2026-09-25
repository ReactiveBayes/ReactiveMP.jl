# Variational only: a `q_γ` contributes the precision E[γ]·I.

@define_message_update_rule(
    node = MvNormalMeanScalePrecision, target = :μ,
    args = (q[:out]::Any, q[:γ]::Any),
    body = (args) -> MvNormalMeanScalePrecision(mean(args.q[:out]), mean(args.q[:γ])),
)

@define_message_update_rule(
    node = MvNormalMeanScalePrecision, target = :μ,
    args = (m[:out]::MultivariateNormalDistributionsFamily, q[:γ]::Any),
    body = (args) -> begin
        μ, V = mean_cov(args.m[:out])
        MvNormalMeanCovariance(μ, V + inv(mean(args.q[:γ])) * diageye(eltype(μ), length(μ)))
    end,
)

# A scale-precision message stays one: precisions l and r in series give lr/(l + r).
@define_message_update_rule(
    node = MvNormalMeanScalePrecision, target = :μ,
    args = (m[:out]::MvNormalMeanScalePrecision, q[:γ]::Any),
    body = (args) -> begin
        l, r = args.m[:out].γ, mean(args.q[:γ])
        MvNormalMeanScalePrecision(mean(args.m[:out]), (l * r) / (l + r))
    end,
)
