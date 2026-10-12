# Towards `out`: Φ of the input, integrated against a normal message, Φ(μ / √(1 + v)).
@define_message_update_rule(
    node = Probit, target = :out, args = (m[:in]::PointMass,),
    body = (args) -> Bernoulli(normcdf(mean(args.m[:in]))),
)

@define_message_update_rule(
    node = Probit, target = :out, args = (m[:in]::UnivariateNormalDistributionsFamily,),
    body = (args) -> Bernoulli(normcdf(mean(args.m[:in]) / sqrt(1 + var(args.m[:in])))),
)

# The rules without expectation propagation, for a model that runs Probit under
# `DefaultAlgorithm()`: the factorisation decides what they read, messages in a joint cluster or
# marginals under mean-field, and the rule towards `in` does not read its own edge.
@define_message_update_rule(
    node = Probit, target = :out, algorithm = DefaultAlgorithm, args = (m[:in]::PointMass,),
    body = (args) -> Bernoulli(normcdf(mean(args.m[:in]))),
)

@define_message_update_rule(
    node = Probit, target = :out, algorithm = DefaultAlgorithm, args = (m[:in]::UnivariateNormalDistributionsFamily,),
    body = (args) -> Bernoulli(normcdf(mean(args.m[:in]) / sqrt(1 + var(args.m[:in])))),
)

@define_message_update_rule(
    node = Probit, target = :out, algorithm = DefaultAlgorithm, args = (q[:in]::PointMass,),
    body = (args) -> Bernoulli(normcdf(mean(args.q[:in]))),
)
