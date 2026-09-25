# An InverseWishart message with scale E[S], which enters log InverseWishart linearly, and
# E[ν] degrees of freedom.

@define_message_update_rule(
    node = InverseWishart, target = :out,
    args = (m[:ν]::PointMass, m[:S]::PointMass),
    logscale = 0,
    body = (args) -> InverseWishartFast(mean(args.m[:ν]), mean(args.m[:S])),
)

@define_message_update_rule(
    node = InverseWishart, target = :out,
    args = (q[:ν]::Any, m[:S]::PointMass),
    body = (args) -> InverseWishartFast(mean(args.q[:ν]), mean(args.m[:S])),
)

@define_message_update_rule(
    node = InverseWishart, target = :out,
    args = (m[:ν]::PointMass, q[:S]::Any),
    body = (args) -> InverseWishartFast(mean(args.m[:ν]), mean(args.q[:S])),
)

@define_message_update_rule(
    node = InverseWishart, target = :out,
    args = (q[:ν]::Any, q[:S]::Any),
    body = (args) -> InverseWishartFast(mean(args.q[:ν]), mean(args.q[:S])),
)
