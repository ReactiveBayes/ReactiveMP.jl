@rule(
    formtype    => typeof(+),
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_in1::Dirac{T}, m_in2::Dirac{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Dirac(mean(m_in1) + mean(m_in2))
    end
)

@rule(
    formtype    => typeof(+),
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_in1::NormalMeanVariance{T}, m_in2::NormalMeanVariance{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return NormalMeanVariance(mean(m_in1) + mean(m_in2), var(m_in1) + var(m_in2))
    end
)

@rule(
    formtype    => typeof(+),
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_in1::NormalMeanVariance{T}, m_in2::Dirac{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return NormalMeanVariance(mean(m_in1) + mean(m_in2), var(m_in1))
    end
)

@rule(
    formtype    => typeof(+),
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_in1::Dirac{T}, m_in2::NormalMeanVariance{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return NormalMeanVariance(mean(m_in1) + mean(m_in2), var(m_in2))
    end
)