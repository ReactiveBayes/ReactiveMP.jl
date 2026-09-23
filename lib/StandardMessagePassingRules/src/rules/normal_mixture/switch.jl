# Each component's weight is the exponentiated negative energy of its own
# NormalMeanPrecision (or MvNormalMeanPrecision) node, normalised.
@define_message_update_rule(
    node = NormalMixture, target = :switch,
    args = (q[:out]::Any, q[:m...]::Any, q[:p...]::Any),
    body = (args) -> begin
        U = [-mixture_component_energy(args.q[:out], m, p) for (m, p) in zip(args.q[:m], args.q[:p])]
        Categorical(clamp!(softmax!(U), tiny, one(eltype(U)) - tiny))
    end,
)
