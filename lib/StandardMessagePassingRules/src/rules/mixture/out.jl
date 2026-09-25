# Towards `out`: the inputs' messages as a mixture, each weighted by its probability and its log
# scale, with the log-sum-exp of those as the message's own.
@define_message_update_rule(
    node = Mixture, algorithm = MixtureBP, target = :out,
    args = (m[:switch]::Any, m[:inputs...]::Any),
    reads_logscale = true, logscale = from_body,
    body = (args) -> begin
        switch = incoming_logscale(args.logscale.m[:switch]) .+ log.(probvec(args.m[:switch]))
        logscales = [incoming_logscale(l) for l in args.logscale.m[:inputs]] .+ switch
        with_logscale(MixtureDistribution(collect(args.m[:inputs]), softmax(logscales)), logsumexp(logscales))
    end,
)
