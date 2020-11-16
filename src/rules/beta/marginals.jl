@marginalrule(
    formtype    => Beta,
    on          => :out_a_b,
    messages    => (m_out::Beta{T}, m_a::Dirac{T}, m_b::Dirac{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return (prod(ProdPreserveParametrisation(), Beta(mean(m_a), mean(m_b)), m_out), m_a, m_b)
    end
)