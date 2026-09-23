# Towards `out`: the inputs' messages as a mixture, each weighted by its probability and its log
# scale, with the log-sum-exp of those as the message's own.
@define_message_update_rule(
    node = Mixture, target = :out,
    args = (m[:switch]::Any, m[:inputs...]::Any),
    body = (args, ann) -> begin
        switch = incoming_logscale(ann.m[:switch]) .+ log.(probvec(args.m[:switch]))
        logscales = [incoming_logscale(a) for a in ann.m[:inputs]] .+ switch
        annotate!(ann, :logscale, logsumexp(logscales))
        MixtureDistribution(collect(args.m[:inputs]), softmax(logscales))
    end,
)
