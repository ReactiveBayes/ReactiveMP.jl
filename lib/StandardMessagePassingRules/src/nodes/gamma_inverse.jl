@define_factor_node(node = GammaInverse, type = Stochastic, interfaces = [:out, (:α, aliases = [:shape]), (:θ, aliases = [:scale])])

# E[θ/x] = θ·E[1/x]; θ/E[x] would be wrong for any q_out that is not a point mass.
@define_average_energy(
    node = GammaInverse,
    args = (q[:out]::GammaInverse, q[:α]::PointMass, q[:θ]::PointMass),
    body = (args) -> begin
        α, θ = mean(args.q[:α]), mean(args.q[:θ])
        -α * log(θ) + loggamma(α) + (α + 1) * mean(log, args.q[:out]) + θ * mean(inv, args.q[:out])
    end,
)
