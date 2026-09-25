@define_message_update_rule(
    node = Poisson, target = :out,
    args = (m[:l]::PointMass,),
    logscale = 0,
    body = (args) -> Poisson(mean(args.m[:l])),
)

@define_message_update_rule(
    node = Poisson, target = :out,
    args = (q[:l]::GammaDistributionsFamily,),
    body = (args) -> Poisson(exp(digamma(shape(args.q[:l]))) / rate(args.q[:l])),
)
