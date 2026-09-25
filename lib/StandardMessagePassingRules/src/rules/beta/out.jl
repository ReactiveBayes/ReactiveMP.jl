@define_message_update_rule(
    node = Beta, target = :out,
    args = (m[:a]::PointMass, m[:b]::PointMass),
    logscale = 0,
    body = (args) -> Beta(mean(args.m[:a]), mean(args.m[:b])),
)

@define_message_update_rule(
    node = Beta, target = :out,
    args = (q[:a]::PointMass, q[:b]::PointMass),
    logscale = 0,
    body = (args) -> Beta(mean(args.q[:a]), mean(args.q[:b])),
)
