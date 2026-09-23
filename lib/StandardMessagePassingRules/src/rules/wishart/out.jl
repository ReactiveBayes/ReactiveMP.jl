# A Wishart message with inverse scale E[S⁻¹]: `mean(cholinv, q_S)` for a `q_S`, which v6
# took as E[S]⁻¹ (ReactiveMP.jl#675); the degrees of freedom enter linearly, as E[ν].

@define_message_update_rule(
    node = Wishart, target = :out,
    args = (m[:ν]::PointMass, m[:S]::PointMass),
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
