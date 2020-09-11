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
    messages    => (m_out::Normal{T}, m_in1::Normal{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Normal(mean(m_out) - mean(m_in1), sqrt(var(m_out) + var(m_in1)))
    end
)

@rule(
    form        => typeof(+),
    on          => :in2,
    vconstraint => Marginalisation,
    messages    => (m_out::Dirac{T}, m_in1::Normal{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Normal(mean(m_out) - mean(m_in1), std(m_in1))
    end
)

@rule(
    form        => typeof(+),
    on          => :in2,
    vconstraint => Marginalisation,
    messages    => (m_out::Normal{T}, m_in1::Dirac{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Normal(mean(m_out) - mean(m_in1), std(m_out))
    end
)