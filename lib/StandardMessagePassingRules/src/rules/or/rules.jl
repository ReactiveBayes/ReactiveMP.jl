@define_message_update_rule(
    node = OR, target = :out,
    args = (m[:in1]::Bernoulli, m[:in2]::Bernoulli),
    logscale = 0,
    body = (args) -> begin
        pin1, pin2 = mean(args.m[:in1]), mean(args.m[:in2])
        Bernoulli(pin1 + pin2 - pin1 * pin2)
    end,
)

# OR is symmetric, as AND is. Unnormalised, the message is pout pother + (1 - pout)(1 - pother)
# at false and pout at true; their sum is its normaliser.
or_input_normaliser(pout, pother) = 1 - pother + 2 * pother * pout
or_towards_input(pout, pother) = Bernoulli(pout / or_input_normaliser(pout, pother))

@define_message_update_rule(
    node = OR, target = :in1,
    args = (m[:out]::Bernoulli, m[:in2]::Bernoulli),
    logscale = (args) -> log(or_input_normaliser(mean(args.m[:out]), mean(args.m[:in2]))),
    body = (args) -> or_towards_input(mean(args.m[:out]), mean(args.m[:in2])),
)

@define_message_update_rule(
    node = OR, target = :in2,
    args = (m[:out]::Bernoulli, m[:in1]::Bernoulli),
    logscale = (args) -> log(or_input_normaliser(mean(args.m[:out]), mean(args.m[:in1]))),
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
