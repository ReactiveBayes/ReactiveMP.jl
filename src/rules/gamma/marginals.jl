@marginalrule(
    formtype    => Gamma,
    on          => :out_α_θ,
    messages    => (m_out::Gamma{T}, m_α::Dirac{T}, m_θ::Dirac{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return (prod(ProdPreserveParametrisation(), Gamma(mean(m_α), mean(m_θ)), m_out), m_α, m_θ)
    end
)