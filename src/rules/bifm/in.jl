
@rule BIFM(:in, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_zprev::MarginalDistribution{<:MultivariateNormalDistributionsFamily}, m_znext::MultivariateNormalDistributionsFamily, meta::BIFMMeta) = begin

    # fetch information of meta data
    A                   = getA(meta)
    B                   = getB(meta)
    μu                  = getμu(meta)
    Σu                  = getΣu(meta)
    ξz                  = getξz(meta)
    Wz                  = getWz(meta)

    # fetch statistic of incoming messages
    mean_znext, V_znext = mean_cov(m_zprev)
    ξz2, Wz2            = weightedmean_precision(m_zprev)

    # calculate intermediate variables
    ξztilde = Wz * mean_znext - ξz
    Wztilde = Wz - Wz * V_znext * Wz

    # calculate marginals of input
    mean_in = μu - Σu * B' * ξztilde
    V_in = Σu - Σu * B' * Wztilde * B * Σu

    # return input marginal
    return MarginalDistribution(MvNormalMeanCovariance(mean_in, V_in))
end