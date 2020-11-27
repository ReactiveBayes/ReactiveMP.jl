@marginalrule(
    formtype    => Gamma,
    on          => :out_α_θ,
    messages    => (m_out::Gamma{T}, m_α::Dirac{T}, m_θ::Dirac{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return (out = prod(ProdPreserveParametrisation(), Gamma(mean(m_α), mean(m_θ)), m_out), α = m_α, θ = m_θ)
    end
)