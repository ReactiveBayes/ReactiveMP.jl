const INTEGER_SWITCH = "Cannot handle switch with Integer values. The switch variable should be a one-hot encoded vector where each element represents the probability of being in that state. Please convert your integer switch values to one-hot encoded vectors before using this rule."

@define_message_update_rule(
    node = NormalMixture, target = (:m, k),
    args = (q[:out]::Any, q[:switch]::Union{Categorical, Bernoulli, PointMass{<:AbstractVector}}, q[:p][k]::Any),
    body = (args) -> begin
        pv = probvec(args.q[:switch])
        z = clamp(pv[k], tiny, one(eltype(pv)) - tiny)
        normal = promote_variate_type(variate_form(typeof(args.q[:out])), NormalMeanPrecision)
        normal(mean(args.q[:out]), z * mean(args.q[:p][k]))
    end,
)

@define_message_update_rule(
    node = NormalMixture, target = (:m, k),
    args = (q[:out]::Any, q[:switch]::PointMass{<:Real}, q[:p][k]::Any),
    body = (args) -> error(INTEGER_SWITCH),
)
