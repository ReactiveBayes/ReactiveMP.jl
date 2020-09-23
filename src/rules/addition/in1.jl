@rule(
    form        => typeof(+),
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
    form        => typeof(+),
    on          => :in1,
    vconstraint => Marginalisation,
    messages    => (m_out::Normal{T}, m_in2::Normal{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Normal(mean(m_out) - mean(m_in2), sqrt(var(m_out) + var(m_in2)))
    end
)

@rule(
    form        => typeof(+),
    on          => :in1,
    vconstraint => Marginalisation,
    messages    => (m_out::Dirac{T}, m_in2::Normal{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Normal(mean(m_out) - mean(m_in2), std(m_in2))
    end
)

@rule(
    form        => typeof(+),
    on          => :in1,
    vconstraint => Marginalisation,
    messages    => (m_out::Normal{T}, m_in2::Dirac{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Normal(mean(m_out) - mean(m_in2), std(m_out))
    end
)