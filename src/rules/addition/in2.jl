@rule(
    form        => typeof(+),
    on          => :in2,
    vconstraint => Marginalisation,
    messages    => (m_out::Dirac{T}, m_in1::Dirac{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Dirac(mean(m_out) - mean(m_in1))
    end
)

@rule(
    form        => typeof(+),
    on          => :in2,
    vconstraint => Marginalisation,
    messages    => (m_out::NormalMeanVariance{T}, m_in1::NormalMeanVariance{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return NormalMeanVariance(mean(m_out) - mean(m_in1), var(m_out) + var(m_in1))
    end
)

@rule(
    form        => typeof(+),
    on          => :in2,
    vconstraint => Marginalisation,
    messages    => (m_out::Dirac{T}, m_in1::NormalMeanVariance{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return NormalMeanVariance(mean(m_out) - mean(m_in1), var(m_in1))
    end
)

@rule(
    form        => typeof(+),
    on          => :in2,
    vconstraint => Marginalisation,
    messages    => (m_out::NormalMeanVariance{T}, m_in1::Dirac{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return NormalMeanVariance(mean(m_out) - mean(m_in1), var(m_out))
    end
)