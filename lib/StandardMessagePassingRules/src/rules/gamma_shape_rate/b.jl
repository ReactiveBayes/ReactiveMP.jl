@define_message_update_rule(
    node = GammaShapeRate, target = :β,
    args = (q[:out]::Any, q[:α]::Any),
    body = (args) -> GammaShapeRate(1 + mean(args.q[:α]), mean(args.q[:out])),
)
