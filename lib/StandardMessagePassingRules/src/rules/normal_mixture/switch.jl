# Each component's weight is the exponentiated negative energy of its own
# NormalMeanPrecision node, normalised.
@define_message_update_rule(
    node = NormalMixture, target = :switch,
    args = (q[:out]::Any, q[:m...]::Any, q[:p...]::Any),
    body = (args) -> begin
        U = [-normal_mean_precision_energy(args.q[:out], m, p) for (m, p) in zip(args.q[:m], args.q[:p])]
        Categorical(clamp!(softmax!(U), tiny, one(eltype(U)) - tiny))
    end,
)
