
@rule BIFM(:znext, Marginalisation) (
    m_out::MultivariateNormalDistributionsFamily,
    m_in::MultivariateNormalDistributionsFamily,
    m_zprev::TerminalProdArgument{<:MultivariateNormalDistributionsFamily},
    meta::BIFMMeta
) = begin

    # fetch information from meta data
    A       = getA(meta)
    B       = getB(meta)
    H       = getH(meta)
    ξztilde = getξztilde(meta)
    BHBt    = getBHBt(meta)
    Λz      = getΛz(meta)

    # fetch statistics of incoming messages
    μ_in, Σ_in       = mean_cov(m_in)
    μ_zprev, Σ_zprev = mean_cov(m_zprev.argument)

    # calculate intermediate quantities
    F = I - Λz * BHBt

    m_ztilde = A * μ_zprev
    V_ztilde = A * Σ_zprev * A'

    # save required intermediate variables
    setΣu!(meta, Σ_in)
    setμu!(meta, μ_in)

    # calculate statistics of outgoing marginal
    μ_znext = F' * m_ztilde + B * ((Σ_in * (B' * ξztilde)) + μ_in)
    Σ_znext = F' * V_ztilde * F + BHBt

    # Actual return type depends on meta object as well, so we explicitly cast the result here
    # Should be noop if type matches
    T = promote_samplefloattype(m_out, m_in, m_zprev.argument)

    # return outgoing marginal
    return TerminalProdArgument(convert(MvNormalMeanCovariance{T}, MvNormalMeanCovariance(μ_znext, Σ_znext)))
end
