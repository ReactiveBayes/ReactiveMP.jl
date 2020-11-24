@rule(
    formtype    => typeof(*),
    on          => :in,
    vconstraint => Marginalisation,
    messages    => (m_out::Dirac, m_A::Dirac),
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Dirac(mean(m_in1) * mean(m_in2))
    end
)

@rule(
    formtype    => typeof(*),
    on          => :in,
    vconstraint => Marginalisation,
    messages    => (m_out::Dist, m_A::Dirac) where { Dist <: Union{ NormalMeanVariance, MvNormalMeanCovariance } },
    marginals   => Nothing,
    meta        => Nothing,
    begin
        A_inv = cholinv(mean(m_A))
        m = mean(m_out)
        P = cov(m_out)
        return Dist(A_inv * m, A_inv * P * A_inv')
    end
)