@rule Switch((:inputs, k), Marginalisation) (m_out::Any, m_switch::Any, m_inputs::ManyOf{N, Any}) where {N} = begin
    @logscale getlogscale(messages[1]) + getlogscale(messages[2]) + log(probvec(messages[2])[k])
    return m_out
end

@rule Switch((:inputs, k), Marginalisation) (m_out::Any, m_switch::Any, m_inputs::Any) = begin
    @logscale getlogscale(messages[1]) + getlogscale(messages[2]) + log(probvec(messages[2])[k])
    return m_out
end
