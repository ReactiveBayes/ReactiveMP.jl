# A Gamma likelihood of the rate, weighted by the responsibility π: shape 1 + π E[a], rate
# π E[out]. Not clamped.
@define_message_update_rule(
    node = GammaMixture, target = (:b, k),
    args = (q[:out]::Any, q[:switch]::Union{Categorical, Bernoulli, PointMass{<:AbstractVector}}, q[:a][k]::Any),
    body = (args) -> begin
        π = probvec(args.q[:switch])[k]
        GammaShapeRate(1 + π * mean(args.q[:a][k]), π * mean(args.q[:out]))
    end,
)

@define_message_update_rule(
    node = GammaMixture, target = (:b, k),
    args = (q[:out]::Any, q[:switch]::PointMass{<:Real}, q[:a][k]::Any),
    body = (args) -> error(INTEGER_SWITCH),
)
