export marginalrule

@marginalrule typeof(dot)(:in1_in2) (m_out::NormalMeanPrecision, m_in1::PointMass, m_in2::MvNormalDistributionsFamily) = begin
    a = mean(m_in1)
    m, V = mean(m_out), cov(m_out)
    x = mean(m_in1)

    q_in2 = prod(ProdPreserveParametrisation(), m_in2, MvNormalWeightedMeanPrecision(x * weightedmean(m_out), x * precision(m_out) * x'))
    return (in1 = a, in2 = q_in2)
end
