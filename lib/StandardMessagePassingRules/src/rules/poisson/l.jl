@define_message_update_rule(
    node = Poisson, target = :l,
    args = (m[:out]::PointMass,),
    body = (args) -> Gamma(mean(args.m[:out]) + 1, 1),
)

@define_message_update_rule(
    node = Poisson, target = :l,
    args = (q[:out]::Any,),
    body = (args) -> Gamma(mean(args.q[:out]) + 1, 1),
)
