export rule

@rule typeof(dot)(:in2, Marginalisation) (m_out::NormalMeanPrecision, m_in1::PointMass) = begin
    x = mean(m_in1)
    d = length(x)
    xi = x * var(m_out) * mean(m_out)
    w = x * var(m_out) * x'
    return NormalMeanPrecision(inv(w)*xi, w)
end
