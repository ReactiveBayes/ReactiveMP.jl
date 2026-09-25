# Belief propagation: the variance's likelihood from the difference of `out` and `μ`, a
# log-density on the half line with no closed-form family.

@define_message_update_rule(
    node = NormalMeanVariance, target = :v,
    args = (m[:out]::PointMass, m[:μ]::UnivariateNormalDistributionsFamily),
    body = (args) -> variance_likelihood(mean(args.m[:out]), mean_var(args.m[:μ])...),
)

@define_message_update_rule(
    node = NormalMeanVariance, target = :v,
    args = (m[:out]::UnivariateNormalDistributionsFamily, m[:μ]::PointMass),
    body = (args) -> variance_likelihood(mean(args.m[:μ]), mean_var(args.m[:out])...),
)

@define_message_update_rule(
    node = NormalMeanVariance, target = :v,
    args = (m[:out]::UnivariateNormalDistributionsFamily, m[:μ]::UnivariateNormalDistributionsFamily),
    body = (args) -> begin
        out_mean, out_var = mean_var(args.m[:out])
        μ_mean, μ_var = mean_var(args.m[:μ])
        variance_likelihood(out_mean, μ_mean, out_var + μ_var)
    end,
)

# p(v) ∝ N(a; b, s + v), the likelihood of the variance of a normal with extra variance `s`.
variance_likelihood(a, b, s) =
    ContinuousUnivariateLogPdf(DomainSets.HalfLine(), (x) -> -log(s + x) / 2 - (a - b)^2 / (2 * (s + x)))

# Variational: an inverse gamma with shape -1/2, built without checking its arguments; the
# scale is half the expected squared difference of `out` and `μ`.

@define_message_update_rule(
    node = NormalMeanVariance, target = :v,
    args = (q[:out]::Any, q[:μ]::Any),
    body = (args) -> begin
        μ_mean, μ_var = mean_var(args.q[:μ])
        out_mean, out_var = mean_var(args.q[:out])
        θ = (out_mean^2 + out_var - 2 * out_mean * μ_mean + μ_mean^2 + μ_var) / 2
        GammaInverse(convert(typeof(θ), -0.5), θ; check_args = false)
    end,
)

@define_message_update_rule(
    node = NormalMeanVariance, target = :v,
    args = (q[:out, :μ]::MultivariateNormalDistributionsFamily,),
    body = (args) -> begin
        m, V = mean_cov(args.q[:out, :μ])
        θ = ((m[1] - m[2])^2 + V[1, 1] - 2 * V[1, 2] + V[2, 2]) / 2
        GammaInverse(convert(typeof(θ), -0.5), θ; check_args = false)
    end,
)
