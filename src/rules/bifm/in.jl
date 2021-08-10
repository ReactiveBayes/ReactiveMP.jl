
@rule BIFM(:in, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_zprev::MarginalDistribution{<:MultivariateNormalDistributionsFamily}, m_znext::MultivariateNormalDistributionsFamily, meta::BIFMMeta) = begin

    # fetch information of meta data
    A                   = getA(meta)
    B                   = getB(meta)
    μu                  = getμu(meta)
    Σu                  = getΣu(meta)
    ξz                  = getξz(meta)
    Λz                  = getΛz(meta)

    # fetch statistic of incoming messages
    μ_znext, Σ_znext    = mean_cov(m_zprev)

    # calculate intermediate variables
    ξztilde = Λz * μ_znext - ξz
    Λztilde = Λz * (I - Σ_znext * Λz)
    tmp = B * Σu

    # calculate marginals of input
    μ_in = μu - (Σu * (B' * ξztilde))
    Σ_in = Σu - tmp' * Λztilde * tmp

    # return input marginal
    return MarginalDistribution(MvNormalMeanCovariance(μ_in, Σ_in))

end