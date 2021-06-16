export rule

@rule BIFM(:input, Marginalisation) (m_output::MvNormalWeightedMeanPrecision, m_zprev::MarginalDistribution{MvNormalMeanCovariance}, m_znext::MvNormalWeightedMeanPrecision, meta::BIFMMeta) = begin
    # todo: optimize for speed

    # fetch statistics
    μu, Σu = getμu(meta), getΣu(meta)
    mean_znext, V_znext = mean_cov(m_znext)
    ξz, Wz = weightedmean_precision(m_zprev)
    B = getB(meta)

    # calculate intermediate variables
    ξztilde = Wz * mean_znext - ξz
    Wztilde = Wz - Wz * V_znext * Wz

    # calculate marginals of input
    mean_input = μu - Σu * B.T * ξztilde
    V_input = Σu - Σu * B.T * Wztilde * B * Σu

    # return input marginal
    return MarginalDistribution(MvNormalMeanCovariance(mean_input, V_input))
end
