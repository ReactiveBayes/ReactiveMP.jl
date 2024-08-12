## VMP
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

## Sum-Product

@rule Binomial(:k, Marginalisation) (m_n::PointMass, m_p::PointMass) = Binomial(BayesBase.getpointmass(m_n), BayesBase.getpointmass(m_p))

@rule Binomial(:k, Marginalisation) (m_n::PointMass, m_p::Any) = begin
    n = BayesBase.getpointmass(m_n)
    supp = DomainSets.ClosedInterval(0:n)
    grid = 0:0.01:1
    logμ_k = (k) -> mapreduce(p -> pdf(Binomial(n,p),k)*pdf(m_p, p), +, grid) - log(length(grid))

    return DiscreteUnivariateLogPdf(supp, logμ_k)
end

@rule Binomial(:k, Marginalisation) (m_n::Any, m_p::Any) = begin
    supp_n = if typeof(m_n) <: ExponentialFamilyDistribution
        DomainSets.Interval(minimum(getsupport(m_n)), maximum(getsupport(m_n)))
    else
        DomainSets.Interval(minimum(Distributions.support(m_n)), maximum(Distributions.support(m_n)))
    end

    supp = DomainSets.ClosedInterval(0:maximum(supp_n))
    grid = 0:0.01:1
    μ_tilde = (k, p) -> mapreduce(n ->  pdf(Binomial(n, p), k) * pdf(m_n, n), +, k:maximum(supp_n))
    logμ_k = (k) -> log(mapreduce(p -> μ_tilde(k, p)*pdf(m_p, p), +, grid)) - log(length(grid))

    return DiscreteUnivariateLogPdf(supp, logμ_k)
end
