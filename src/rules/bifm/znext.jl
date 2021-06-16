export rule

@rule BIFM(:znext, Marginalisation) (m_output::MvNormalWeightedMeanPrecision, m_input::MvNormalWeightedMeanPrecision, m_zprev::MarginalDistribution{MvNormalMeanCovariance}, meta::BIFMMeta) = begin
    # todo: optimize for speed

    # fetch statistics
    μ_u, V_u = mean_cov(m_input)
    ξ_output, W_output = weightedmean_precision(m_output)
    μ_zprev, V_zprev = mean_cov(m_zprev)
    A, B, H, ξ_ztilde, Wz = getA(meta), getB(meta), getH(meta), getξztilde(meta), getWz(meta)

    # calculate intermediate quantities
    F = I - Wz * B * H * B.T

    m_ztilde = A * μ_zprev
    V_ztilde = A * V_zprev * A.T

    m_znext = F.T * m_ztilde + B * V_u * B.T * ξ_ztilde + B * μ_u
    V_znext = F.T * V_ztilde * F + B * H * B.T

    # return outgoing marginal
    return MarginalDistribution(MvNormalMeanCovariance(m_znext, V_znext))

end
