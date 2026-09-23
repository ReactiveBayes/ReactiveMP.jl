# The shape's likelihood, exp(γα - loggamma(α)) with γ = E[log β] + E[log out].
@define_message_update_rule(
    node = GammaShapeRate, target = :α,
    args = (q[:out]::Any, q[:β]::GammaDistributionsFamily),
    body = (args) -> GammaShapeLikelihood(promote(1, mean(log, args.q[:β]) + mean(log, args.q[:out]))...),
)
