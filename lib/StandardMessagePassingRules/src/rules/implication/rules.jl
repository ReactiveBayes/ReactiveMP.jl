@define_message_update_rule(
    node = IMPLY, target = :out,
    args = (m[:in1]::Bernoulli, m[:in2]::Bernoulli),
    logscale = 0,
    body = (args) -> begin
        pin1, pin2 = mean(args.m[:in1]), mean(args.m[:in2])
        Bernoulli(1 - pin1 + pin1 * pin2)
    end,
)

# Unnormalised, the message towards `in1` is pout at false and pout pin2 + (1 - pout)(1 - pin2) at
# true, and the one towards `in2` is pout (1 - pin1) + (1 - pout) pin1 at false and pout at true;
# their sums are their normalisers.
imply_in1_normaliser(pout, pin2) = 1 - pin2 + 2 * pout * pin2
imply_in2_normaliser(pout, pin1) = 2 * pout + pin1 - 2 * pout * pin1

@define_message_update_rule(
    node = IMPLY, target = :in1,
    args = (m[:out]::Bernoulli, m[:in2]::Bernoulli),
    logscale = (args) -> log(imply_in1_normaliser(mean(args.m[:out]), mean(args.m[:in2]))),
    body = (args) -> begin
        pout, pin2 = mean(args.m[:out]), mean(args.m[:in2])
        Bernoulli((1 - pout - pin2 + 2 * pout * pin2) / imply_in1_normaliser(pout, pin2))
    end,
)

@define_message_update_rule(
    node = IMPLY, target = :in2,
    args = (m[:out]::Bernoulli, m[:in1]::Bernoulli),
    logscale = (args) -> log(imply_in2_normaliser(mean(args.m[:out]), mean(args.m[:in1]))),
    body = (args) -> begin
        pout, pin1 = mean(args.m[:out]), mean(args.m[:in1])
        Bernoulli(pout / imply_in2_normaliser(pout, pin1))
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
