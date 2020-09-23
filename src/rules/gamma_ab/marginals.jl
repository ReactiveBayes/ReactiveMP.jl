@marginalrule(
    form        => Type{ <: GammaAB },
    on          => :out_a_b,
    messages    => (m_out::GammaAB{T}, m_a::Dirac{T}, m_b::Dirac{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        q_out = Message(GammaAB(mean(m_a), mean(m_b))) * m_out
        return (getdata(q_out), getdata(m_a), getdata(m_b))
    end
)