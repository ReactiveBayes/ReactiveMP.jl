export rule

@rule BIFM(:input, Marginalisation) (m_input::MvNormalWeightedMeanPrecision, m_znext::Marginal{MvNormalMeanCovariance, meta::BIFMMeta) = begin
    # todo: optimize for speed

    # fetch statistics
    μu, Σu = mean_cov(m_input)
    mean_znext, V_znext = mean_cov(m_znext)
    ξz, Wz = weightedmean_precision(m_zprev)
    B = getB(meta)

    # calculate intermediate variables
    ξtildez = Wz * mean_znext - ξz
    Wtildez = Wz - Wz * V_znext * Wz

    # calculate marginals of input
    mean_input = μu - Σu * B.T * ξtildez
    V_input = Σu - Σu * B.T * Wtildez * B * Σu

    # return input marginal
    return MvNormalMeanCovariance(mean_input, V_input)
end
