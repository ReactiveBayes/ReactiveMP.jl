export rule

# TODO: NormalWeightedMeanPrecision
@rule typeof(dot)(:in1, Marginalisation) (m_out::NormalMeanPrecision, m_in2::PointMass) = begin
    x = mean(m_in2)
    d = length(x)
    xi = x * precision(m_out) * mean(m_out)
    w = x * precision(m_out) * x'
    return MvNormalMeanPrecision(inv(w)*xi, w)
end
