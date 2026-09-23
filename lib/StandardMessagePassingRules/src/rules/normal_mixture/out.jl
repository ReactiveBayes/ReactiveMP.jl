@define_message_update_rule(
    node = NormalMixture, target = :out,
    args = (q[:switch]::Any, q[:m...]::Any, q[:p...]::Any),
    body = (args) -> begin
        πs = probvec(args.q[:switch])
        precisions = map(mean, args.q[:p])
        means = map(mean, args.q[:m])
        W = sum(k -> πs[k] * precisions[k], eachindex(precisions))
        ξ = sum(k -> πs[k] * precisions[k] * means[k], eachindex(precisions))
        NormalWeightedMeanPrecision(ξ, W)
    end,
)
