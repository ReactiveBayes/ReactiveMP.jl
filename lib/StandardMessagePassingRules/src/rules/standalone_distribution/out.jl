@define_message_update_rule(
    node = StandaloneDistribution, target = :out, args = (q[:distribution]::PointMass{<:Distribution},),
    logscale = 0,
    body = (args) -> BayesBase.getpointmass(args.q[:distribution]),
)
