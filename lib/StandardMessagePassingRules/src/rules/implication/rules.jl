@define_message_update_rule(
    node = IMPLY, target = :out,
    args = (m[:in1]::Bernoulli, m[:in2]::Bernoulli),
    body = (args) -> begin
        pin1, pin2 = mean(args.m[:in1]), mean(args.m[:in2])
        Bernoulli(1 - pin1 + pin1 * pin2)
    end,
)

@define_message_update_rule(
    node = IMPLY, target = :in1,
    args = (m[:out]::Bernoulli, m[:in2]::Bernoulli),
    body = (args) -> begin
        pout, pin2 = mean(args.m[:out]), mean(args.m[:in2])
        Bernoulli((1 - pout - pin2 + 2 * pout * pin2) / (1 - pin2 + 2 * pout * pin2))
    end,
)

@define_message_update_rule(
    node = IMPLY, target = :in2,
    args = (m[:out]::Bernoulli, m[:in1]::Bernoulli),
    body = (args) -> begin
        pout, pin1 = mean(args.m[:out]), mean(args.m[:in1])
        Bernoulli(pout / (2 * pout + pin1 - 2 * pout * pin1))
    end,
)

@define_marginal_update_rule(
    node = IMPLY, target = (:in1, :in2),
    args = (m[:out]::Bernoulli, m[:in1]::Bernoulli, m[:in2]::Bernoulli),
    body = (args) -> begin
        pin1, pin2, pout = mean(args.m[:in1]), mean(args.m[:in2]), mean(args.m[:out])
        Contingency([(1 - pin1) * pout * (1 - pin2) (1 - pin1) * pin2 * pout; pin1 * (1 - pin2) * (1 - pout) pin1 * pin2 * pout])
    end,
)
