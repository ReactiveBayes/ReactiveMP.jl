@rule Binomial(:p, Marginalisation) (q_k::Any, q_n::Any) = begin
    m_nk = mean(q_n) - mean(q_k)
    return m_nk ≥ 0 ? Beta(mean(q_k) + 1, m_nk + 1) : Beta(mean(q_k) + 1, 1)
end
@rule Binomial(:p, Marginalisation) (m_k::PointMass, m_n::PointMass) = begin
    m_nk = mean(m_n) - mean(m_k)
    return m_nk ≥ 0 ? Beta(mean(m_k) + 1, m_nk + 1) : Beta(mean(m_k) + 1, 1)
end
@rule Binomial(:p, Marginalisation) (m_k::Any, m_n::Any) = begin
    μ_tilde = (n, p) -> sum(map(k -> pdf(Binomial(n, p), k) * pdf(m_k, k), Distributions.support(m_k)))
    logμ_p = (p) -> logsumexp(map(n -> log(μ_tilde(n, p)) + logpdf(m_n, n), Distributions.support(m_n)))

    return ContinuousUnivariateLogPdf(DomainSets.ClosedInterval(0.0, 1.0), logμ_p)
end
