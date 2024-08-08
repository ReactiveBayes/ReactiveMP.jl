using LogExpFunctions
@rule Binomial(:n, Marginalisation) (q_k::Any, q_p::Any) = begin
    basemeasure = n -> ReactiveMP.expectation_logbinomial(q_k, PointMass(n))
    ss = (identity,)
    η = (mean(mirrorlog, q_p),)
    lb = if typeof(q_k) <: PointMass
        BayesBase.getpointmass(q_k)
    elseif typeof(q_k) <: ExponentialFamilyDistribution
        maximum(getsupport(q_k))
    else
        maximum(Distributions.support(q_k))
    end
    supp = lb:Int(lb + floor(logfactorial(lb)))
    logpartition = (η) -> logsumexp(map(x -> mapreduce((f, θ) -> θ' * f(x) + log(basemeasure(x)), collect, ss, η), supp))
    attributes = ExponentialFamilyDistributionAttributes(basemeasure, ss, logpartition, supp)
    ef = ExponentialFamilyDistribution(Univariate, η, nothing, attributes)

    return ef
end

@rule Binomial(:n, Marginalisation) (m_k::Any, m_p::Any) = begin
    lb = if typeof(m_k) <: PointMass
        BayesBase.getpointmass(m_k)
    elseif typeof(m_k) <: ExponentialFamilyDistribution
        maximum(getsupport(m_k))
    else
        maximum(Distributions.support(m_k))
    end
    μ_tilde = (n, p) -> sum(map(k -> pdf(Binomial(n, p), k) * pdf(m_k, k), 0:lb))
    logμ_n = (n) -> logsumexp(map(p -> log(μ_tilde(n, p)), 0:0.001:1))
    return DiscreteUnivariateLogPdf(DomainSets.Interval(lb, Int(lb + floor(logfactorial(lb)))), logμ_n)
end
