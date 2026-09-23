@define_factor_node(node = Bernoulli, type = Stochastic, interfaces = [:out, (:p, aliases = [:θ])])

@define_average_energy(
    node = Bernoulli,
    args = (q[:out]::Any, q[:p]::Any),
    body = (args) -> -mean(args.q[:out]) * mean(log, args.q[:p]) - (1 - mean(args.q[:out])) * mean(mirrorlog, args.q[:p]),
)
