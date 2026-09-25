# Towards `switch`: component k's evidence is the log scale of m_out × m_inputs[k], which is the
# product's own (`product_logscale`, with the algorithm's product strategy) plus the two incoming
# ones.
@define_message_update_rule(
    node = Mixture, algorithm = MixtureBP, target = :switch,
    args = (m[:out]::Any, m[:inputs...]::Any),
    reads_logscale = true, logscale = from_body,
    body = (algo, args) -> begin
        out = incoming_logscale(args.logscale.m[:out])
        logscales = [out + incoming_logscale(l) + product_logscale(algo.prod, args.m[:out], m) for (m, l) in zip(args.m[:inputs], args.logscale.m[:inputs])]
        with_logscale(Categorical(softmax(logscales)), logsumexp(logscales))
    end,
)
