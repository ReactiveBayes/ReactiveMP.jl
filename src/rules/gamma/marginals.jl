export marginalrule

@marginalrule Gamma(:out_α_θ) (m_out::Gamma, m_α::PointMass, m_θ::PointMass) = begin
    return (out = prod(ClosedProd(), Gamma(mean(m_α), mean(m_θ)), m_out), α = m_α, θ = m_θ)
end
