export rule

@rule typeof(dot)(:in2, Marginalisation) (m_out::NormalMeanPrecision, m_in1::PointMass) = begin
    x = mean(m_in1)
    d = length(x)
    xi = x * precision(m_out) * mean(m_out)
    w = x * precision(m_out) * x'
    return MvNormalMeanPrecision(inv(w)*xi, w)
end
