# Towards `switch`: component k's evidence is the log scale of m_out × m_inputs[k], which is the
# product's own (`ctx.product`) plus the two incoming ones.
@define_message_update_rule(
    node = Mixture, target = :switch, ctx = (:product,),
    args = (m[:out]::Any, m[:inputs...]::Any),
    body = (ctx, args, ann) -> begin
        out = incoming_logscale(ann.m[:out])
        logscales = [out + incoming_logscale(a) + last(ctx.product(args.m[:out], m)) for (m, a) in zip(args.m[:inputs], ann.m[:inputs])]
        annotate!(ann, :logscale, logsumexp(logscales))
        Categorical(softmax(logscales))
    end,
)
