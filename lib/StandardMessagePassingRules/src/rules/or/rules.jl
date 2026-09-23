@define_message_update_rule(
    node = OR, target = :out,
    args = (m[:in1]::Bernoulli, m[:in2]::Bernoulli),
    body = (args) -> begin
        pin1, pin2 = mean(args.m[:in1]), mean(args.m[:in2])
        Bernoulli(pin1 + pin2 - pin1 * pin2)
    end,
)

# OR is symmetric, as AND is.
or_towards_input(pout, pother) = Bernoulli(pout / (1 - pother + 2 * pother * pout))

@define_message_update_rule(
    node = OR, target = :in1,
    args = (m[:out]::Bernoulli, m[:in2]::Bernoulli),
    body = (args) -> or_towards_input(mean(args.m[:out]), mean(args.m[:in2])),
)

@define_message_update_rule(
    node = OR, target = :in2,
    args = (m[:out]::Bernoulli, m[:in1]::Bernoulli),
    body = (args) -> or_towards_input(mean(args.m[:out]), mean(args.m[:in1])),
)

@define_marginal_update_rule(
    node = OR, target = (:in1, :in2),
    args = (m[:out]::Bernoulli, m[:in1]::Bernoulli, m[:in2]::Bernoulli),
    body = (args) -> begin
        pin1, pin2, pout = mean(args.m[:in1]), mean(args.m[:in2]), mean(args.m[:out])
        Contingency([(1 - pin1) * (1 - pin2) * (1 - pout) (1 - pin1) * pin2 * pout; pin1 * (1 - pin2) * pout pin1 * pin2 * pout])
    end,
)
