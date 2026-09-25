@define_message_update_rule(
    node = NOT, target = :out,
    args = (m[:in]::Bernoulli,),
    logscale = 0,
    body = (args) -> Bernoulli(1 - mean(args.m[:in])),
)

@define_message_update_rule(
    node = NOT, target = :in,
    args = (m[:out]::Bernoulli,),
    logscale = 0,
    body = (args) -> Bernoulli(1 - mean(args.m[:out])),
)
