@rule(
    form        => Type{ <: InverseWishart },
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_ν::Dirac, m_Ψ::Dirac),
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return InverseWishart(mean(m_v), mean(m_Ψ))
    end
)