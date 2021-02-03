export rule

@rule typeof(dot)(:in1, Marginalisation) (m_out::NormalMeanPrecision, m_in2::PointMass) = begin
    x = mean(m_in2)
    d = length(x)
    xi = x * var(m_out) * mean(m_out)
    w = x * var(m_out) * x'
    return NormalMeanPrecision(inv(w)*xi, w)
end
