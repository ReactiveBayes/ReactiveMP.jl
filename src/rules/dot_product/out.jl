export rule

@rule typeof(dot)(:out, Marginalisation, symmetrical=[:in1, :in2]) (m_in1::PointMass, m_in2::MultivariateNormalDistributionsFamily) = begin
    NormalMeanVariance(mean(m_in1)'*mean(m_in2), mean(m_in1)'*var(m_in2)*mean(m_in1))
end
