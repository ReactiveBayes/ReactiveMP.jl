@define_factor_node(
    node = GammaShapeRate,
    type = Stochastic,
    interfaces = [:out, (:α, aliases = [:a, :shape]), (:β, aliases = [:b, :rate])],
)

gamma_shape_rate_energy(q_out, q_α, q_β) =
    mean(loggamma, q_α) - mean(q_α) * mean(log, q_β) - (mean(q_α) - 1) * mean(log, q_out) + mean(q_β) * mean(q_out)

@define_average_energy(
    node = GammaShapeRate,
    args = (q[:out]::Any, q[:α]::PointMass, q[:β]::Any),
    body = (args) -> gamma_shape_rate_energy(args.q[:out], args.q[:α], args.q[:β]),
)

@define_average_energy(
    node = GammaShapeRate,
    args = (q[:out]::Any, q[:α]::GammaDistributionsFamily, q[:β]::Any),
    body = (args) -> gamma_shape_rate_energy(args.q[:out], args.q[:α], args.q[:β]),
)
