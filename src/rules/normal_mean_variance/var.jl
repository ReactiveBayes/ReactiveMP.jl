
# Belief Propagation                #
# --------------------------------- #
@rule NormalMeanVariance(:v, Marginalisation) (m_out::PointMass, m_μ::UnivariateNormalDistributionsFamily) = begin
    m_out_mean        = mean(m_out)
    m_μ_mean, m_μ_cov = mean_cov(m_μ)

    return ContinuousUnivariateLogPdf(
        DomainSets.HalfLine(),
        (x) -> -log(m_μ_cov + x) / 2 - 1 / (2 * (x + m_μ_cov)) * (m_out_mean - m_μ_mean)^2
    )
end

@rule NormalMeanVariance(:v, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_μ::PointMass) = begin
    m_out_mean, m_out_cov = mean_cov(m_out)
    m_μ_mean              = mean(m_μ)

    return ContinuousUnivariateLogPdf(
        DomainSets.HalfLine(),
        (x) -> -log(m_out_cov + x) / 2 - 1 / (2 * (x + m_out_cov)) * (m_out_mean - m_μ_mean)^2
    )
end

@rule NormalMeanVariance(:v, Marginalisation) (
    m_out::UnivariateNormalDistributionsFamily,
    m_μ::UnivariateNormalDistributionsFamily
) = begin
    m_out_mean, m_out_cov = mean_cov(m_out)
    m_μ_mean, m_μ_cov     = mean_cov(m_μ)

    return ContinuousUnivariateLogPdf(
        DomainSets.HalfLine(),
        (x) -> -log(m_out_cov + m_μ_cov + x) / 2 - 1 / (2 * (m_out_cov + m_μ_cov + x)) * (m_out_mean - m_μ_mean)^2
    )
end
