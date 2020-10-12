@rule(
    form        => Type{ <: Wishart },
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_ν::Dirac, m_S::Dirac),
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Wishart(mean(m_v), mean(m_S))
    end
)