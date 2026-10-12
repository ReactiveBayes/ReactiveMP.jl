# Towards `in`, expectation propagation: the tilted distribution's moments over the cavity.
@define_message_update_rule(
    node = Probit, target = :in, args = (m[:out]::Union{PointMass, Bernoulli}, m[:in]::UnivariateNormalDistributionsFamily),
    args_check = (args) -> check_probability(args.m[:out]),
    body = (args) -> begin
        mpz, vpz, mz, vz = tilted_moments(mean(args.m[:out]), args.m[:in])
        NormalWeightedMeanPrecision(mpz / vpz - mz / vz, clamp(1 / vpz - 1 / vz, tiny, huge))
    end,
)

# The likelihood of `in` given an output probability p: log(1 - p + (2p - 1) Φ(z)).
probit_likelihood(p) = ContinuousUnivariateLogPdf(z -> log(1 - p + (2 * p - 1) * normcdf(z)))

@define_message_update_rule(
    node = Probit, target = :in, algorithm = DefaultAlgorithm, args = (m[:out]::Union{PointMass, Bernoulli},),
    body = (args) -> probit_likelihood(mean(args.m[:out])),
)

@define_message_update_rule(
    node = Probit, target = :in, algorithm = DefaultAlgorithm, args = (q[:out]::PointMass,),
    body = (args) -> probit_likelihood(mean(args.q[:out])),
)
