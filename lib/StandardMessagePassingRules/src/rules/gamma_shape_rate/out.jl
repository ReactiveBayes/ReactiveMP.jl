@define_message_update_rule(
    node = GammaShapeRate, target = :out,
    args = (m[:α]::PointMass, m[:β]::PointMass),
    logscale = 0,
    body = (args) -> GammaShapeRate(mean(args.m[:α]), mean(args.m[:β])),
)

@define_message_update_rule(
    node = GammaShapeRate, target = :out,
    args = (q[:α]::Any, q[:β]::Any),
    body = (args) -> GammaShapeRate(mean(args.q[:α]), mean(args.q[:β])),
)
