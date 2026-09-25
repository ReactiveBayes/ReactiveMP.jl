# Each component's weight is the exponentiated negative energy of its own GammaShapeRate node,
# clamped away from 0 and 1 and renormalised.
@define_message_update_rule(
    node = GammaMixture, target = :switch,
    args = (q[:out]::Any, q[:a...]::Any, q[:b...]::GammaDistributionsFamily),
    body = (args) -> begin
        U = [-gamma_shape_rate_energy(args.q[:out], a, b) for (a, b) in zip(args.q[:a], args.q[:b])]
        ρ = clamp.(softmax(U), tiny, one(eltype(U)) - tiny)
        Categorical(ρ ./ sum(ρ))
    end,
)
