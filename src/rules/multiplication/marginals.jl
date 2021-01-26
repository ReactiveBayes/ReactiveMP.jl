export marginalrule

@marginalrule typeof(*)(:A_in) (m_out::MvNormalMeanCovariance, m_A::PointMass, m_in::MvNormalMeanCovariance) = begin
    # TODO with @call_rule macro or smth
    A_inv = cholinv(mean(m_A))
    m = mean(m_out)
    P = cov(m_out)
    q_in = prod(ProdPreserveParametrisation(), m_in, MvNormalMeanCovariance(A_inv * m, A_inv * P * A_inv'))
    return (A = m_A, in = q_in)
end