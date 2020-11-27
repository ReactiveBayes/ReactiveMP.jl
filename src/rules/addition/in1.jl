@rule(
    formtype    => typeof(+),
    on          => :in1,
    vconstraint => Marginalisation,
    messages    => (m_out::Dirac{T}, m_in2::Dirac{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Dirac(mean(m_out) - mean(m_in2))
    end
)

@rule(
    formtype    => typeof(+),
    on          => :in1,
    vconstraint => Marginalisation,
    messages    => (m_out::NormalMeanVariance{T}, m_in2::Normal{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return NormalMeanVariance(mean(m_out) - mean(m_in2), var(m_out) + var(m_in2))
    end
)

@rule(
    formtype    => typeof(+),
    on          => :in1,
    vconstraint => Marginalisation,
    messages    => (m_out::Dirac{T}, m_in2::NormalMeanVariance{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return NormalMeanVariance(mean(m_out) - mean(m_in2), var(m_in2))
    end
)

@rule(
    formtype    => typeof(+),
    on          => :in1,
    vconstraint => Marginalisation,
    messages    => (m_out::NormalMeanVariance{T}, m_in2::Dirac{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return NormalMeanVariance(mean(m_out) - mean(m_in2), var(m_out))
    end
)

@rule(
    formtype    => typeof(+),
    on          => :in1,
    vconstraint => Marginalisation,
    messages    => (m_out::Dirac{T}, m_in2::NormalMeanPrecision{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return NormalMeanPrecision(mean(m_out) - mean(m_in2), precision(m_in2))
    end
)

@rule(
    formtype    => typeof(+),
    on          => :in1,
    vconstraint => Marginalisation,
    messages    => (m_out::NormalMeanPrecision{T}, m_in2::NormalMeanPrecision{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        p1, p2 = precision(m_out), precision(m_in2)
        return NormalMeanPrecision(mean(m_out) - mean(m_in2), p1 * p2 / (p1 + p2))
    end
)