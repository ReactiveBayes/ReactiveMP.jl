export rule

@rule BIFM(:output, Marginalisation) (m_input::MultivariateNormalDistributionsFamily, m_zprev::MarginalDistribution{<:MultivariateNormalDistributionsFamily}, m_znext::MultivariateNormalDistributionsFamily, meta::BIFMMeta) = begin
    # todo: optimize for speed

    # fetch statistics
    μ_u, V_u = mean_cov(m_input)
    μ_zprev, V_zprev = mean_cov(m_zprev)
    A, B, C, H, ξ_ztilde, Wz = getA(meta), getB(meta), getC(meta), getH(meta), getξztilde(meta), getWz(meta)

    # calculate intermediate quantities
    F = I - Wz * B * H * B'

    m_ztilde = A * μ_zprev
    V_ztilde = A * V_zprev * A'

    m_znext = F' * m_ztilde + B * V_u * B' * ξ_ztilde + B * μ_u
    V_znext = F' * V_ztilde * F + B * H * B'

    # calculate outgoing message to output
    m_output = C * m_znext
    V_output = C * V_znext * C'

    # return outgoing marginal
    return MarginalDistribution(MvNormalMeanCovariance(m_output, V_output))
end
