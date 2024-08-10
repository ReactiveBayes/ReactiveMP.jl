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

@rule Binomial(:n, Marginalisation) (m_k::PointMass, m_p::PointMass) = begin
    k = BayesBase.getpointmass(m_k)
    p = BayesBase.getpointmass(m_p)

    return DiscreteUnivariateLogPdf(DomainSets.ClosedInterval(k:Int(k + floor(logfactorial(k)))), n -> logpdf(Binomial(n, p),k))
end

@rule Binomial(:n, Marginalisation) (m_k::PointMass, m_p::Any) = begin
    k = BayesBase.getpointmass(m_k)
    grid = 0:0.01:1
    logμ_tilde = (n, p) -> logpdf(Binomial(n, p), k)
    logμ_n = (n) -> logsumexp(map(p -> logμ_tilde(n, p) == -Inf || logpdf(m_p, p) == -Inf ? 0 : logμ_tilde(n, p) + logpdf(m_p, p), grid))

    return DiscreteUnivariateLogPdf(DomainSets.ClosedInterval(k:Int(k + floor(logfactorial(k)))), logμ_n)
end

@rule Binomial(:n, Marginalisation) (m_k::Any, m_p::Any) = begin
    ub = if typeof(m_k) <: ExponentialFamilyDistribution
        maximum(getsupport(m_k))
    else
        maximum(Distributions.support(m_k))
    end
    grid = 0:0.01:1
    logμ_tilde = (n, p) -> logsumexp(map(k -> logpdf(Binomial(n, p), k) == -Inf || logpdf(m_k, k) == -Inf ? 0 : logpdf(Binomial(n, p), k) + logpdf(m_k, k), 0:ub))
    logμ_n = (n) -> logsumexp(map(p -> logμ_tilde(n, p) == -Inf || logpdf(m_p, p) == -Inf ? 0 : logμ_tilde(n, p) + logpdf(m_p, p), grid))

    return DiscreteUnivariateLogPdf(DomainSets.ClosedInterval(ub:Int(ub + floor(logfactorial(ub)))), logμ_n)
end
