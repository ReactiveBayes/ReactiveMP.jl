@marginalrule GammaInverse(:out_α_θ) (m_out::GammaInverse, m_α::PointMass, m_θ::PointMass) = begin
    return (
        out = prod(
            ProdAnalytical(),
            GammaInverse(mean(m_α), mean(m_θ)),
            m_out
        ),
        α = m_α,
        θ = m_θ
    )
end
