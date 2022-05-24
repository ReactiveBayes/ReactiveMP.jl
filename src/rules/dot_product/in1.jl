
@rule typeof(dot)(:in1, Marginalisation) (
    m_out::UnivariateNormalDistributionsFamily,
    m_in2::PointMass,
    meta::AbstractCorrection
) = begin
    return @call_rule typeof(dot)(:in2, Marginalisation) (m_out = m_out, m_in1 = m_in2, meta = meta)
end
