export rule

@rule BIFM(:input, Marginalisation) (m_output::MultivariateNormalDistributionsFamily, m_zprev::MarginalDistribution{<:MultivariateNormalDistributionsFamily}, m_znext::MultivariateNormalDistributionsFamily, meta::BIFMMeta) = begin
    # todo: optimize for speed

    # fetch statistics
    A, B                = getA(meta), getB(meta)
    μu, Σu              = getμu(meta), getΣu(meta)
    mean_znext, V_znext = mean_cov(m_zprev)
#    ξz, Wz              = weightedmean_precision(m_znext)
    ξz                  = getξz(meta)
    Wz                  = getWz(meta)

    ξz2, Wz2            = weightedmean_precision(m_zprev)#

    println(A, B, μu, Σu, mean_znext, V_znext, ξz, Wz, ξz2, Wz2)
    # calculate intermediate variables
    ξztilde = Wz * mean_znext - ξz
    Wztilde = Wz - Wz * V_znext * Wz

    # calculate marginals of input
    mean_input = μu - Σu * B' * ξztilde
    V_input = Σu - Σu * B' * Wztilde * B * Σu

    # return input marginal
    return MarginalDistribution(MvNormalMeanCovariance(mean_input, V_input))
end
