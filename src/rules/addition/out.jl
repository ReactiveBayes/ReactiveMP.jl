@rule(
    form        => typeof(+),
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
    form        => typeof(+),
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_in1::Normal{T}, m_in2::Normal{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Normal(mean(m_in1) + mean(m_in2), sqrt(var(m_in1) + var(m_in2)))
    end
)

@rule(
    form        => typeof(+),
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_in1::Normal{T}, m_in2::Dirac{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Normal(mean(m_in1) + mean(m_in2), std(m_in1))
    end
)

@rule(
    form        => typeof(+),
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_in1::Dirac{T}, m_in2::Normal{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Normal(mean(m_in1) + mean(m_in2), std(m_in2))
    end
)