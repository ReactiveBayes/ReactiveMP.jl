# The responsibilities' mixture of the components' likelihoods: shape Σ π E[a], since Σ π = 1
# makes Σ π (E[a] - 1) + 1 the same, and rate Σ π E[b].
@define_message_update_rule(
    node = GammaMixture, target = :out,
    args = (q[:switch]::Any, q[:a...]::Any, q[:b...]::GammaDistributionsFamily),
    body = (args) -> begin
        πs = probvec(args.q[:switch])
        GammaShapeRate(sum(k -> πs[k] * mean(args.q[:a][k]), eachindex(πs)), sum(k -> πs[k] * mean(args.q[:b][k]), eachindex(πs)))
    end,
)
