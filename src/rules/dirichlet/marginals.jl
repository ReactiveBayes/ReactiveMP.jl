@marginalrule(
    formtype    => Dirichlet,
    on          => :out_a,
    messages    => (m_out::Dirichlet, m_a::Dirac),
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return (out = prod(ProdPreserveParametrisation(), Dirichlet(mean(m_a)), m_out), a = m_a)
    end
)