@define_factor_node(node = GammaInverse, type = Stochastic, interfaces = [:out, (:α, aliases = [:shape]), (:θ, aliases = [:scale])])

# The support is x > 0, so E[θ/x] = θ E[1/x] is defined; v6 wrote it θ / E[x].
@define_average_energy(
    node = GammaInverse,
    args = (q[:out]::GammaInverse, q[:α]::PointMass, q[:θ]::PointMass),
    body = (args) -> begin
        α, θ = mean(args.q[:α]), mean(args.q[:θ])
        -α * log(θ) + loggamma(α) + (α + 1) * mean(log, args.q[:out]) + θ / mean(args.q[:out])
    end,
)
