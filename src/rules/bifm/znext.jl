export rule

@rule BIFM(:znext, Marginalisation) (m_output::MultivariateNormalDistributionsFamily, m_input::MultivariateNormalDistributionsFamily, m_zprev::MarginalDistribution{<:MultivariateNormalDistributionsFamily}, meta::BIFMMeta) = begin
    # todo: optimize for speed

    # fetch statistics
    μ_u, V_u = mean_cov(m_input)
    ξ_output, W_output = weightedmean_precision(m_output)
    μ_zprev, V_zprev = mean_cov(m_zprev)
    A, B, H, ξ_ztilde, Wz = getA(meta), getB(meta), getH(meta), getξztilde(meta), getWz(meta)

    # calculate intermediate quantities
    F = I - Wz * B * H * B'

    m_ztilde = A * μ_zprev
    V_ztilde = A * V_zprev * A'

    m_znext = F' * m_ztilde + B * V_u * B' * ξ_ztilde + B * μ_u
    V_znext = F' * V_ztilde * F + B * H * B'

    # return outgoing marginal
    return MarginalDistribution(MvNormalMeanCovariance(m_znext, V_znext))

end
