@rule typeof(-)(:in1, Marginalisation) (m_out::Any, m_in2::Any, meta::Any) = begin
    @call_rule typeof(+)(:out, Marginalisation) (m_in1 = m_out, m_in2 = m_in2, meta = meta)
end
