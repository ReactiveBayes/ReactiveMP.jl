@define_message_update_rule(
    node = StandaloneDistribution, target = :out, args = (q[:distribution]::PointMass{<:Distribution},),
    body = (args) -> BayesBase.getpointmass(args.q[:distribution]),
)
