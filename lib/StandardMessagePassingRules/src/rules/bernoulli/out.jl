@define_message_update_rule(
    node = Bernoulli, target = :out,
    args = (m[:p]::Beta,),
    body = (args, ann) -> begin
        annotate!(ann, :logscale, 0)
        Bernoulli(mean(args.m[:p]))
    end,
)

@define_message_update_rule(
    node = Bernoulli, target = :out,
    args = (m[:p]::PointMass,),
    body = (args) -> Bernoulli(mean(args.m[:p])),
)

@define_message_update_rule(
    node = Bernoulli, target = :out,
    args = (q[:p]::PointMass,),
    body = (args) -> Bernoulli(mean(args.q[:p])),
)

# exp(E[log p]) against exp(E[log(1 - p)]), normalised and clamped to [tiny, 1].
@define_message_update_rule(
    node = Bernoulli, target = :out,
    args = (q[:p]::Any,),
    body = (args) -> begin
        rho_1, rho_2 = mean(log, args.q[:p]), mean(mirrorlog, args.q[:p])
        m = max(rho_1, rho_2)
        tmp = exp(rho_1 - m)
        Bernoulli(clamp(tmp / (tmp + exp(rho_2 - m)), tiny, one(m)))
    end,
)
