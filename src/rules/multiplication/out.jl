@rule(
    formtype    => typeof(*),
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_A::Dirac, m_in::Dirac),
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Dirac(mean(m_in1) * mean(m_in2))
    end
)

@rule(
    formtype    => typeof(*),
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_A::Dirac, m_in::Dist) where { Dist <: Union{ NormalMeanVariance, MvNormalMeanCovariance } },
    marginals   => Nothing,
    meta        => Nothing,
    begin
        A = mean(m_A)
        m = mean(m_in)
        P = cov(m_in)
        return Dist(A * m, A * P * A')
    end
)