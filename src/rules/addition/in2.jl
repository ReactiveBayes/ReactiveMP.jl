@rule(
    formtype    => typeof(+),
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
    formtype    => typeof(+),
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
    formtype    => typeof(+),
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
    formtype    => typeof(+),
    on          => :in2,
    vconstraint => Marginalisation,
    messages    => (m_out::NormalMeanVariance{T}, m_in1::Dirac{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return NormalMeanVariance(mean(m_out) - mean(m_in1), var(m_out))
    end
)

@rule(
    formtype    => typeof(+),
    on          => :in2,
    vconstraint => Marginalisation,
    messages    => (m_out::Dirac{T}, m_in1::NormalMeanPrecision{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return NormalMeanPrecision(mean(m_out) - mean(m_in1), precision(m_in1))
    end
)

@rule(
    formtype    => typeof(+),
    on          => :in2,
    vconstraint => Marginalisation,
    messages    => (m_out::NormalMeanPrecision{T}, m_in1::NormalMeanPrecision{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        p1, p2 = precision(m_out), precision(m_in1)
        return NormalMeanPrecision(mean(m_out) - mean(m_in1), p1 * p2 / (p1 + p2))
    end
)