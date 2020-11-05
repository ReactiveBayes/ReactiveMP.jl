@marginalrule(
    formtype    => Wishart,
    on          => :out_ν_S,
    messages    => (m_out::Wishart, m_ν::Dirac, m_S::Dirac),
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return (prod(ProdPreserveParametrisation(), Wishart(mean(m_ν), mean(m_S)), m_out), m_ν, m_S)
    end
)