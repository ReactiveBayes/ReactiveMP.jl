# Towards input k: the message from `out`, scaled by the probability of k and the two incoming
# log scales.
@define_message_update_rule(
    node = Mixture, algorithm = MixtureBP, target = (:inputs, k),
    args = (m[:out]::Any, m[:switch]::Any),
    reads_logscale = true,
    logscale = (args) -> incoming_logscale(args.logscale.m[:out]) + incoming_logscale(args.logscale.m[:switch]) + log(probvec(args.m[:switch])[k]),
    body = (args) -> args.m[:out],
)
