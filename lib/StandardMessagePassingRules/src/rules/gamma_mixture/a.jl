# The likelihood of the shape, weighted by the component's responsibility p: exp(γ a - p log Γ(a))
# with γ = p (E[log out] + E[log b]). Not clamped.
@define_message_update_rule(
    node = GammaMixture, target = (:a, k),
    args = (q[:out]::Any, q[:switch]::Union{Categorical, Bernoulli, PointMass{<:AbstractVector}}, q[:b][k]::GammaDistributionsFamily),
    body = (args) -> begin
        p = probvec(args.q[:switch])[k]
        GammaShapeLikelihood(p, p * (mean(log, args.q[:out]) + mean(log, args.q[:b][k])))
    end,
)

@define_message_update_rule(
    node = GammaMixture, target = (:a, k),
    args = (q[:out]::Any, q[:switch]::PointMass{<:Real}, q[:b][k]::GammaDistributionsFamily),
    body = (args) -> error(INTEGER_SWITCH),
)
