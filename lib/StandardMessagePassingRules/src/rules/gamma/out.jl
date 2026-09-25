@define_message_update_rule(
    node = Gamma, target = :out,
    args = (m[:α]::PointMass, m[:θ]::PointMass),
    logscale = 0,
    body = (args) -> Gamma(mean(args.m[:α]), mean(args.m[:θ])),
)

@define_message_update_rule(
    node = Gamma, target = :out,
    args = (q[:α]::Any, q[:θ]::Any),
    body = (args) -> Gamma(mean(args.q[:α]), mean(args.q[:θ])),
)
