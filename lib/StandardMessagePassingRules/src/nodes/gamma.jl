@define_factor_node(node = Gamma, type = Stochastic, interfaces = [:out, (:α, aliases = [:shape]), (:θ, aliases = [:scale])])

gamma_energy(q_out, q_α, q_θ) =
    mean(loggamma, q_α) + mean(q_α) * mean(log, q_θ) - (mean(q_α) - 1) * mean(log, q_out) + mean(q_out) / mean(q_θ)

@define_average_energy(
    node = Gamma,
    args = (q[:out]::Any, q[:α]::PointMass, q[:θ]::Any),
    body = (args) -> gamma_energy(args.q[:out], args.q[:α], args.q[:θ]),
)

@define_average_energy(
    node = Gamma,
    args = (q[:out]::Any, q[:α]::GammaDistributionsFamily, q[:θ]::Any),
    body = (args) -> gamma_energy(args.q[:out], args.q[:α], args.q[:θ]),
)
