export rule

@rule BIFM(:znext, Marginalisation) (m_output::MvNormalWeightedMeanPrecision, m_input::MvNormalWeightedMeanPrecision, m_zprev::Marginal{MvNormalWeightedMeanPrecision}, meta::BIFMMeta) = begin 
    # todo: optimize for speed

    # fetch statistics
    ξ_input, W_input = weightedmean_precision(m_input)
    ξ_znext, W_znext = weightedmean_precision(m_znext)
    A, B, C = getA(meta), getB(meta), getC(meta)

    # calculate intermediate quantities
    H = cholinv(W_input + B.T * W_znext * B)
    ξ_ztilde = ξ_znext + W_znext * B * H * (-ξ_input - B.T * ξ_znext)
    W_ztilde = W_znext - W_znext * B * H * B.T * W_znext

    # save required intermediate quantities
    # todo: save H
    # todo: save ξ_ztilde

    # calculate outgoing message to zprev
    ξ_zprev = A.T * ξ_ztilde
    W_zprev = A.T * W_ztilde * A

end