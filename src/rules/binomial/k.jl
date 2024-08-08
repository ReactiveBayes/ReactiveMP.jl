@rule Binomial(:k, Marginalisation) (q_n::PointMass, q_p::Any) = Binomial(mean(q_n), mean(q_p))

@rule Binomial(:k, Marginalisation) (q_n::Binomial, q_p::Any) = begin
    logbasemeasure = k -> log(ReactiveMP.expectation_logbinomial(PointMass(k), q_n))
    basemeasure = k -> ReactiveMP.expectation_logbinomial(PointMass(k), q_n)
    ss = (identity,)
    η = (mean(log, q_p) - mean(mirrorlog, q_p),)
    logpartition = (η) -> logsumexp(map(x -> mapreduce((f, θ) -> θ' * f(x) + logbasemeasure(x), collect, ss, η), Distributions.support(q_n)))
    attributes = ExponentialFamilyDistributionAttributes(basemeasure, ss, logpartition, Distributions.support(q_n))
    ef = ExponentialFamilyDistribution(Univariate, η, nothing, attributes)
    return ef
end

@rule Binomial(:k, Marginalisation) (m_n::Any, m_p::Any) = begin
    supp = if typeof(m_n) <: PointMass
        DomainSets.Interval(0, BayesBase.getpointmass(m_n))
    elseif typeof(m_n) <: ExponentialFamilyDistribution
        DomainSets.Interval(minimum(getsupport(m_n)), maximum(getsupport(m_n)))
    else
        DomainSets.Interval(minimum(Distributions.support(m_n)), maximum(Distributions.support(m_n)))
    end

    μ_tilde = (k, p) -> sum(map(n -> pdf(Binomial(n, p), k) * pdf(m_n, n), minimum(supp):maximum(supp)))
    logμ_k = (k) -> logsumexp(map(p -> log(μ_tilde(k, p)), 0:0.001:1))
    return DiscreteUnivariateLogPdf(supp, logμ_k)
end
