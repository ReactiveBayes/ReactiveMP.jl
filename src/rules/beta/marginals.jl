@marginalrule(
    formtype    => Beta,
    on          => :out_a_b,
    messages    => (m_out::Beta{T}, m_a::Dirac{T}, m_b::Dirac{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return (out = prod(ProdPreserveParametrisation(), Beta(mean(m_a), mean(m_b)), m_out), a = m_a, b = m_b)
    end
)