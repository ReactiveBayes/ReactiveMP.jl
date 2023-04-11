
@marginalrule typeof(dot)(:in1_in2) (m_out::NormalDistributionsFamily, m_in1::PointMass, m_in2::NormalDistributionsFamily, meta::AbstractCorrection) = begin

    # Forward message towards `in2` edge
    mf_in2 = @call_rule typeof(dot)(:in2, Marginalisation) (m_out = m_out, m_in1 = m_in1, meta = meta)
    q_in2  = prod(ProdAnalytical(), m_in2, mf_in2)

    return convert_paramfloattype((in1 = m_in1, in2 = q_in2))
end

@marginalrule typeof(dot)(:in1_in2) (m_out::NormalDistributionsFamily, m_in1::NormalDistributionsFamily, m_in2::PointMass, meta::AbstractCorrection) = begin
    symmetric = @call_marginalrule typeof(dot)(:in1_in2) (m_out = m_out, m_in1 = m_in2, m_in2 = m_in1, meta = meta)
    return convert_paramfloattype((in1 = symmetric[:in2], in2 = symmetric[:in1]))
end
