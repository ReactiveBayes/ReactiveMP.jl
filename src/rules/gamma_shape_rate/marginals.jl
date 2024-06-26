export marginalrule

@marginalrule GammaShapeRate(:out_α_β) (m_out::GammaDistributionsFamily, m_α::PointMass, m_β::PointMass) = begin
    return (out = prod(ClosedProd(), GammaShapeRate(mean(m_α), mean(m_β)), m_out), α = m_α, β = m_β)
end
