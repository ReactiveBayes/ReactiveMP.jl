@rule typeof(-)(:out, Marginalisation) (m_in1::Any, m_in2::Any, meta::Any) = begin
    @call_rule typeof(+)(:in1, Marginalisation) (m_out = m_in1, m_in2 = m_in2, meta = meta)
end