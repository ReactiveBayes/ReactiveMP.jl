@rule Binomial(:p, Marginalisation) (q_n::Any, q_k::Any) = begin
    m_nk = mean(q_n) - mean(q_k)
    return m_nk ≥ 0 ? Beta(mean(q_k) + 1, m_nk + 1) : Beta(mean(q_k) + 1, 1)
end
@rule Binomial(:p, Marginalisation) (m_n::PointMass, m_k::PointMass) = begin
    m_nk = mean(m_n) - mean(m_k)
    return m_nk ≥ 0 ? Beta(mean(m_k) + 1, m_nk + 1) : Beta(mean(m_k) + 1, 1)
end
@rule Binomial(:p, Marginalisation) (m_n::Any, m_k::Any) = begin
    logμ_tilde = (n, p) -> logsumexp(map(k -> logpdf(Binomial(n, p), k) == -Inf || logpdf(m_k, k) == -Inf ? 0 : logpdf(Binomial(n, p), k) + logpdf(m_k, k), Distributions.support(m_k)))
    logμ_p     = (p) -> logsumexp(map(n -> logμ_tilde(n, p) == -Inf || logpdf(m_n, n) == -Inf ? 0 : logμ_tilde(n, p) + logpdf(m_n, n), Distributions.support(m_n)))

    return ContinuousUnivariateLogPdf(DomainSets.ClosedInterval(0.0, 1.0), logμ_p)
end
