@define_message_update_rule(
    node = AND, target = :out,
    args = (m[:in1]::Bernoulli, m[:in2]::Bernoulli),
    body = (args) -> Bernoulli(mean(args.m[:in1]) * mean(args.m[:in2])),
)

# AND is symmetric, so the message towards either input is one function of the message from
# `out` and the other input's.
and_towards_input(pout, pother) = Bernoulli((1 - pout - pother + 2 * pout * pother) / (2 - 2 * pout - pother + 2 * pout * pother))

@define_message_update_rule(
    node = AND, target = :in1,
    args = (m[:out]::Bernoulli, m[:in2]::Bernoulli),
    body = (args) -> and_towards_input(mean(args.m[:out]), mean(args.m[:in2])),
)

@define_message_update_rule(
    node = AND, target = :in2,
    args = (m[:out]::Bernoulli, m[:in1]::Bernoulli),
    body = (args) -> and_towards_input(mean(args.m[:out]), mean(args.m[:in1])),
)

@define_marginal_update_rule(
    node = AND, target = (:in1, :in2),
    args = (m[:out]::Bernoulli, m[:in1]::Bernoulli, m[:in2]::Bernoulli),
    body = (args) -> begin
        pin1, pin2, pout = mean(args.m[:in1]), mean(args.m[:in2]), mean(args.m[:out])
        Contingency([(1 - pin1) * (1 - pin2) * (1 - pout) (1 - pin1) * pin2 * (1 - pout); pin1 * (1 - pin2) * (1 - pout) pin1 * pin2 * pout])
    end,
)
