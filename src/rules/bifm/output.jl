export rule

@rule BIFM(:output, Marginalisation) (m_znext::Marginal{MvNormalMeanCovariance, meta::BIFMMeta) = begin
    # todo: optimize for speed

    # fetch statistics
    mean_znext, V_znext = mean_cov(m_znext)
    C = getC(meta)

    # calculate outgoing message to zprev
    mean_output = C * mean_znext
    V_output = C * V_znext * C.T

    # return outgoing marginal
    return MvNormalMeanCovariance(mean_output, V_output)
end
