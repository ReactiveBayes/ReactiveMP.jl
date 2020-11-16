@marginalrule(
    formtype    => Bernoulli,
    on          => :out_p,
    messages    => (m_out::Dirac{T}, m_p::Beta{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        # Check
        return (m_out, prod(ProdPreserveParametrisation(), Beta(one(T) + mean(m_out), 2one(T) - mean(m_out)), m_p))
    end
)