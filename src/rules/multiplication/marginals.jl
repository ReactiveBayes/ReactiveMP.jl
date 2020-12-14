@marginalrule(
    formtype    => typeof(*),
    on          => :A_in,
    messages    => (m_out::MvNormalMeanCovariance, m_A::Dirac, m_in::MvNormalMeanCovariance),
    marginals   => Nothing,
    meta        => Nothing,
    begin
        # TODO with @call_rule macro or smth
        A_inv = cholinv(mean(m_A))
        m = mean(m_out)
        P = cov(m_out)
        q_in = prod(ProdPreserveParametrisation(), m_in, MvNormalMeanCovariance(A_inv * m, A_inv * P * A_inv'))
        return (A = m_A, in = q_in)
    end
)