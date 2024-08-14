using LogExpFunctions
##VMP
@rule Binomial(:n, Marginalisation) (q_k::Any, q_p::Any) = begin
    logbasemeasure = n -> ReactiveMP.expectation_logbinomial(q_k, PointMass(n))
    basemeasure    = n -> exp(ReactiveMP.expectation_logbinomial(q_k, PointMass(n)))
    ss             = (identity,)
    η              = (mean(mirrorlog, q_p),)
    lb             = if typeof(q_k) <: PointMass
        BayesBase.getpointmass(q_k)
    elseif typeof(q_k) <: ExponentialFamilyDistribution
        maximum(getsupport(q_k))
    else
        maximum(Distributions.support(q_k))
    end

    supp         = lb:Int(lb + ceil(logfactorial(lb)))
    logpartition = (η) -> logsumexp(map(x -> mapreduce((f, θ) -> θ' * f(x) , collect, ss, η) + logbasemeasure(x), supp))
    attributes   = ExponentialFamilyDistributionAttributes(basemeasure, ss, logpartition, supp)
    ef           = ExponentialFamilyDistribution(Univariate, η, nothing, attributes)

    return ef
end

##Sum-Product
@rule Binomial(:n, Marginalisation) (m_k::PointMass, m_p::PointMass) = begin
    k = BayesBase.getpointmass(m_k)
    p = BayesBase.getpointmass(m_p)

    return DiscreteUnivariateLogPdf(DomainSets.ClosedInterval(k:Int(k + ceil(logfactorial(k)))), n -> logpdf(Binomial(n, p),k))
end

@rule Binomial(:n, Marginalisation) (m_k::PointMass, m_p::Any) = begin
    k      = BayesBase.getpointmass(m_k)
    grid   = 0:0.01:1
    logμ_n = (n) -> log(mapreduce(p -> pdf(Binomial(n, p), k)* pdf(m_p, p), +, grid)) - log(length(grid))

    return DiscreteUnivariateLogPdf(DomainSets.ClosedInterval(k:Int(k + ceil(logfactorial(k)))), logμ_n)
end

@rule Binomial(:n, Marginalisation) (m_k::Any, m_p::Any) = begin
    ub = if typeof(m_k) <: ExponentialFamilyDistribution
        maximum(getsupport(m_k))
    else
        maximum(Distributions.support(m_k))
    end
    grid    = 0:0.01:1
    μ_tilde = (n, p) -> mapreduce(k -> pdf(Binomial(n,p),k)*pdf(m_k, k) , +, 0:ub )
    logμ_n  = (n) -> log(mapreduce(p -> μ_tilde(n, p)* pdf(m_p, p), + , grid)) - log(length(grid))
    return DiscreteUnivariateLogPdf(DomainSets.ClosedInterval(ub:Int(ub + ceil(logfactorial(ub)))), logμ_n)
end
