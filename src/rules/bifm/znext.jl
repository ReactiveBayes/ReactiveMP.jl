
@rule BIFM(:znext, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in::MultivariateNormalDistributionsFamily, m_zprev::MarginalDistribution{<:MultivariateNormalDistributionsFamily}, meta::BIFMMeta) = begin

    # fetch information from meta data
    A       = getA(meta)
    B       = getB(meta)
    H       = getH(meta)
    ξztilde = getξztilde(meta)
    Wz      = getWz(meta)

    # fetch statistics of incoming messages
    μ_in, Σ_in          = mean_cov(m_in)
    ξ_out, Λ_out        = weightedmean_precision(m_out)
    μ_zprev, Σ_zprev    = mean_cov(m_zprev)

    # calculate intermediate quantities
    F = I - Wz * B * H * B'

    m_ztilde = A * μ_zprev
    V_ztilde = A * V_zprev * A'

    # calculate statistics of outgoing marginal
    μ_znext = F' * m_ztilde + B * V_u * B' * ξ_ztilde + B * μ_u
    Σ_znext = F' * V_ztilde * F + B * H * B'

    # return outgoing marginal
    return MarginalDistribution(MvNormalMeanCovariance(μ_znext, Σ_znext))

end