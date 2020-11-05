@rule(
    formtype    => Wishart,
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_ν::Dirac, m_S::Dirac),
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Wishart(mean(m_ν), mean(m_S))
    end
)

@rule(
    formtype    => Wishart,
    on          => :out,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_ν::Dirac, q_S::Dirac),
    meta        => Nothing,
    begin
        return Wishart(mean(q_ν), mean(q_S))
    end
)