export marginalrule

@marginalrule GammaInverse(:out_α_β) (m_out::GammaInverse, m_α::PointMass, m_β::PointMass) = begin
    return (
        out = prod(
            ProdAnalytical(),
            GammaInverse(mean(m_α), mean(m_β)),
            m_out
        ),
        α = m_α,
        β = m_β
    )
end
