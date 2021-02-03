export marginalrule

@marginalrule typeof(dot)(:out_in2) (m_out::NormalMeanPrecision, m_in1::PointMass, m_in2::MvNormalMeanPrecision) = begin
    m_in1 = mean(m_in1)
    m = mean(m_out)
    P = precision(m_out)
    q_in = prod(ProdPreserveParametrisation(), m_in2, MvNormalMeanPrecision(m_in1 * m, m_in1 * P * m_in1'))
    return (in1 = m_in1, in2 = q_in)
end
