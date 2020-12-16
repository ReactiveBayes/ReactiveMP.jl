export marginalrule

@marginalrule Gamma(:out_α_θ) (m_out::Gamma, m_α::Dirac, m_θ::Dirac) = begin
    return (out = prod(ProdPreserveParametrisation(), Gamma(mean(m_α), mean(m_θ)), m_out), α = m_α, θ = m_θ)
end