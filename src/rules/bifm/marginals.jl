@marginalrule BIFM(:in_zprev_znext) (
    m_out::MultivariateNormalDistributionsFamily,
    m_in::MultivariateNormalDistributionsFamily,
    m_zprev::ProdFinal{<:MultivariateNormalDistributionsFamily},
    m_znext::MultivariateNormalDistributionsFamily,
    meta::BIFMMeta
) = begin
    # Note: this rules is equal to the rule for the :in_zprev marginal. However, it yield identical performance with respect to the RTS smoother (aside from the additional prior on the last z)

    # fetch statistics from meta data
    A = getA(meta)
    B = getB(meta)
    C = getC(meta)
    ξ_ztilde = getξztilde(meta)
    Λ_ztilde = getΛztilde(meta)

    # # extract parameters from messages
    ξ_zprev_marginal, Λ_zprev_marginal = weightedmean_precision(m_zprev)
    ξ_in, Λ_in = weightedmean_precision(m_in)
    ξ_out, Λ_out = weightedmean_precision(m_out)

    # calculate outgoing message of zprev
    ξ_zprev_message = A' * ξ_ztilde
    Λ_zprev_message = A' * Λ_ztilde * A

    # Actual return type depends on meta object as well, so we explicitly cast the result here
    # Should be noop if type matches
    T = promote_samplefloattype(m_out, m_in, m_zprev, m_znext)

    # calculate message towards znext from y
    dist1 = convert(MvNormalWeightedMeanPrecision{T}, MvNormalWeightedMeanPrecision(C' * ξ_out, C' * Λ_out * C))

    # calculate message from z towards the addition node
    dist2 = prod(ProdAnalytical(), dist1, m_znext)
    ξ2, Λ2 = weightedmean_precision(dist2)

    # # calculate joint message from the addition node
    BA = hcat(B, A)'
    ξ3 = convert(AbstractVector{T}, BA * ξ2)
    Λ3 = convert(AbstractMatrix{T}, BA * Λ2 * BA')

    # # create a joint message of the input messages
    ξ4 = convert(AbstractVector{T}, vcat(ξ_in, (ξ_zprev_marginal - ξ_zprev_message)))
    Λ4 = convert(AbstractMatrix{T}, cat(Λ_in, (Λ_zprev_marginal - Λ_zprev_message); dims = (1, 2)))

    # # return joint marginal
    left = MvNormalWeightedMeanPrecision(ξ3, Λ3)
    right = MvNormalWeightedMeanPrecision(ξ4, Λ4)

    return prod(ProdAnalytical(), left, right)
end
