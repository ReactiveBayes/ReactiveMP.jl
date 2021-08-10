
@rule BIFM(:zprev, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in::MultivariateNormalDistributionsFamily, m_znext::MultivariateNormalDistributionsFamily, meta::BIFMMeta) = begin
    # todo: optimize for speed

    # fetch statistics
    ξ_in, W_in = weightedmean_precision(m_in)    # ξu,    Wu
    ξ_out, W_out = weightedmean_precision(m_out) # ξxk,   Wx
    ξ_znext, W_znext = weightedmean_precision(m_znext)    # ξzk+1, Wzk+1
    A, B, C = getA(meta), getB(meta), getC(meta)          # A, B, C

    # calculate intermediate quantities
    ξ_z = C' * ξ_out + ξ_znext
    W_z = C' * W_out * C + W_znext
    H = cholinv(W_in + B' * W_z * B)
    ξ_ztilde = ξ_z + W_z * B * H * (-ξ_in - B' * ξ_z)
    W_ztilde = W_z - W_z * B * H * B' * W_z

    # save required intermediate quantities
    setξz!(meta, ξ_z)
    setH!(meta, H)
    setξztilde!(meta, ξ_ztilde)
    setWz!(meta, W_z)

    # calculate outgoing message to zprev
    ξ_zprev = A' * ξ_ztilde
    W_zprev = A' * W_ztilde * A

    # return message
    return MvNormalWeightedMeanPrecision(ξ_zprev, W_zprev)

end