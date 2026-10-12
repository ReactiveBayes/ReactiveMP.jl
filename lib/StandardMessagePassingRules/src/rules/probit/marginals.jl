# The joint of a point-mass output and `in`: the output, and the tilted distribution of `in`.
@define_marginal_update_rule(
    node = Probit, target = (:out, :in), args = (m[:out]::PointMass, m[:in]::UnivariateNormalDistributionsFamily),
    args_check = (args) -> check_probability(args.m[:out]),
    body = (args) -> begin
        mpz, vpz, _, _ = tilted_moments(mean(args.m[:out]), args.m[:in])
        cluster = FactorizedCluster((:out,) => args.m[:out], (:in,) => NormalMeanVariance(mpz, vpz))
        BayesBase.convert_paramfloattype(BayesBase.promote_paramfloattype(args.m[:out], args.m[:in]), cluster)
    end,
)
