# An observation r ∈ {0, 1} is the likelihood Beta(1 + r, 2 - r) of `p`: p ↦ p or p ↦ 1 - p,
# which integrates to 1/2, hence the log scale log(1/2).
bernoulli_likelihood(r) = Beta(one(r) + r, 2one(r) - r)

@define_message_update_rule(
    node = Bernoulli, target = :p,
    args = (m[:out]::PointMass,),
    logscale = loghalf,
    body = (args) -> bernoulli_likelihood(mean(args.m[:out])),
)

@define_message_update_rule(
    node = Bernoulli, target = :p,
    args = (q[:out]::PointMass,),
    logscale = loghalf,
    body = (args) -> bernoulli_likelihood(mean(args.q[:out])),
)

@define_message_update_rule(
    node = Bernoulli, target = :p,
    args = (q[:out]::Bernoulli,),
    body = (args) -> bernoulli_likelihood(succprob(args.q[:out])),
)

@define_message_update_rule(
    node = Bernoulli, target = :p,
    args = (q[:out]::Categorical,),
    args_check = (args) -> length(probvec(args.q[:out])) == 2 || lazy"`q(out)` must have two categories, the support {0, 1}; got $(length(probvec(args.q[:out])))",
    body = (args) -> bernoulli_likelihood(probvec(args.q[:out])[2]),
)
