# Towards input k: the message from `out`, scaled by the probability of k and the two incoming
# log scales.
@define_message_update_rule(
    node = Mixture, target = (:inputs, k),
    args = (m[:out]::Any, m[:switch]::Any),
    body = (args, ann) -> begin
        annotate!(ann, :logscale, incoming_logscale(ann.m[:out]) + incoming_logscale(ann.m[:switch]) + log(probvec(args.m[:switch])[k]))
        args.m[:out]
    end,
)
