@marginalrule(
    form        => Type{ <: InverseWishart },
    on          => :out_ν_Ψ,
    messages    => (m_out::InverseWishart, m_ν::Dirac, m_Ψ::Dirac),
    marginals   => Nothing,
    meta        => Nothing,
    begin
        q_out = Message(InverseWishart(mean(m_ν), mean(m_Ψ))) * m_out
        return FactorizedMarginal(q_out, m_ν, m_Ψ)
    end
)