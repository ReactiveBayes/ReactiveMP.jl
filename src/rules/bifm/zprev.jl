export rule

# Compute backwards information filter using input, output, and previous backwards information filter

@rule BIFM(:zprev, Marginalisation) (m_output::MvNormalWeightedMeanPrecision, m_input::MvNormalWeightedMeanPrecision, m_znext::MvNormalWeightedMeanPrecision, meta::BIFMMeta) = begin
    # todo: optimize for speed

    # fetch statistics
    ξ_input, W_input = weightedmean_precision(m_input)    # ξu,    Wu
    ξ_output, W_output = weightedmean_precision(m_output) # ξxk,   Wx
    ξ_znext, W_znext = weightedmean_precision(m_znext)    # ξzk+1, Wzk+1
    A, B, C = getA(meta), getB(meta), getC(meta)          # A, B, C

    # calculate intermediate quantities
    H = cholinv(W_input + B.T * W_znext * B)
    ξ_ztilde = ξ_znext + W_znext * B * H * (-ξ_input - B.T * ξ_znext)
    W_ztilde = W_znext - W_znext * B * H * B.T * W_znext

    # save required intermediate quantities
    # save H
    setH!(H)

    # save ξ_ztilde
    setξztilde!(ξ_ztilde)

    # calculated intermediate quantities
    ξ_z1 = A.T * ξ_ztilde
    W_z1 = A.T * W_ztilde * A

    # calculate outgoing message to zprev
    ξ_zprev = C.T * ξ_output + ξ_z1       # ξzk
    W_zprev = C.T * W_output * C + W_z1   # Wzk

    # save Wz
    setWz!(W_zprev)

    # return message
    return MvNormalWeightedMeanPrecision(ξ_zprev, W_zprev)

end
