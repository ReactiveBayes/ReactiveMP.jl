@marginalrule(
    form        => Type{ <: GammaAB },
    on          => :out_a_b,
    messages    => (m_out::GammaAB{T}, m_a::Dirac{T}, m_b::Dirac{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        q_out = Message(GammaAB(mean(m_a), mean(m_b))) * m_out
        return FactorizedMarginal(q_out, m_a, m_b)
    end
)