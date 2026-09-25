@define_message_update_rule(
    node = Dirichlet, target = :out,
    args = (m[:a]::PointMass{<:AbstractVector},),
    logscale = 0,
    body = (args) -> Dirichlet(mean(args.m[:a])),
)

@define_message_update_rule(
    node = Dirichlet, target = :out,
    args = (q[:a]::PointMass{<:AbstractVector},),
    logscale = 0,
    body = (args) -> Dirichlet(mean(args.q[:a])),
)
