
@rule BIFM(:out, Marginalisation) (m_in::MultivariateNormalDistributionsFamily, m_zprev::ProdFinal{<:MultivariateNormalDistributionsFamily}, m_znext::MultivariateNormalDistributionsFamily, meta::BIFMMeta) = begin

    # fetch information from meta data
    A       = getA(meta)
    B       = getB(meta)
    C       = getC(meta)
    H       = getH(meta)
    ξztilde = getξztilde(meta)
    BHBt    = getBHBt(meta)
    Λz      = getΛz(meta)

    # fetch statistics
    μ_in, Σ_in       = mean_cov(m_in)
    μ_zprev, Σ_zprev = mean_cov(m_zprev)

    # calculate intermediate quantities
    F = I - Λz * BHBt

    μ_ztilde = A * μ_zprev
    Σ_ztilde = A * Σ_zprev * A'

    μ_znext = F' * μ_ztilde + (B * (Σ_in * (B' * ξztilde))) + B * μ_in
    Σ_znext = F' * Σ_ztilde * F + BHBt

    # save required intermediate variables
    setΣu!(meta, Σ_in)

    # calculate outgoing message to output
    μ_out = C * μ_znext
    Σ_out = C * Σ_znext * C'

    # Actual return type depends on meta object as well, so we explicitly cast the result here
    # Should be noop if type matches
    T = promote_type(eltype(m_in), eltype(m_zprev), eltype(m_znext))

    # return outgoing marginal
    return ProdFinal(convert(MvNormalMeanCovariance{T}, MvNormalMeanCovariance(μ_out, Σ_out)))

end