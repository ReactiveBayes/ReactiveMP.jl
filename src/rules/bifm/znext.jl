export rule

@rule BIFM(:znext, Marginalisation) (m_output::MvNormalWeightedMeanPrecision, m_input::MvNormalWeightedMeanPrecision, m_zprev::Marginal{MvNormalMeanCovariance}, m_znext::MvNormalWeightedMeanPrecision, meta::BIFMMeta) = begin
    # todo: optimize for speed

    # fetch statistics
    μu, Σu = mean_cov(m_input)
    ξ_output, W_output = weightedmean_precision(m_output)
    mean_zprev, V_prev = mean_cov(m_zprev)
    A, B, H, ξztilde, Wz = getA(meta), getB(meta), getC(meta), getH(meta), getξztilde(meta), getWz(meta)

    # calculate intermediate quantities
    F = I - Wz * B * H * B.T

    mz1 = A * mean_zprev
    Vz1 = A * V_zprev * V.T

    m_znext = F.T * mz1 + B * Σu * B.T * ξ_output + B *  μu
    V_znext = F.T * Vz1 * F + B * H * B.T

    # return outgoing marginal
    return MvNormalMeanCovariance(m_znext, V_znext)

end
