@marginalrule(
    form        => Type{ <: Wishart },
    on          => :out_ν_S,
    messages    => (m_out::Wishart, m_ν::Dirac, m_S::Dirac),
    marginals   => Nothing,
    meta        => Nothing,
    begin
        q_out = Message(Wishart(mean(m_ν), mean(m_S))) * m_out
        return FactorizedMarginal(q_out, m_ν, m_S)
    end
)