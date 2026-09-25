# A Wishart message with inverse scale E[S⁻¹]: `mean(cholinv, q_S)` for a `q_S`, not E[S]⁻¹;
# the degrees of freedom enter linearly, as E[ν].

@define_message_update_rule(
    node = Wishart, target = :out,
    args = (m[:ν]::PointMass, m[:S]::PointMass),
    logscale = 0,
    body = (args) -> WishartFast(mean(args.m[:ν]), cholinv(mean(args.m[:S]))),
)

@define_message_update_rule(
    node = Wishart, target = :out,
    args = (q[:ν]::Any, m[:S]::PointMass),
    body = (args) -> WishartFast(mean(args.q[:ν]), cholinv(mean(args.m[:S]))),
)

@define_message_update_rule(
    node = Wishart, target = :out,
    args = (m[:ν]::PointMass, q[:S]::Any),
    body = (args) -> WishartFast(mean(args.m[:ν]), mean(cholinv, args.q[:S])),
)

@define_message_update_rule(
    node = Wishart, target = :out,
    args = (q[:ν]::Any, q[:S]::Any),
    body = (args) -> WishartFast(mean(args.q[:ν]), mean(cholinv, args.q[:S])),
)
