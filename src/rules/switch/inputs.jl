@rule Switch((:inputs, k), Marginalisation) (
    m_out::Any,
    m_switch::Any,
    m_inputs::Tuple{Any}
) = begin
    @logscale getlogscale(messages[1]) + getlogscale(messages[2]) + log(probvec(messages[2])[k])
    return m_out
end

@rule Switch((:inputs, k), Marginalisation) (
    m_out::Any,
    m_switch::Any,
    m_inputs::Any
) = begin
    @logscale getlogscale(messages[1]) + getlogscale(messages[2]) + log(probvec(messages[2])[k])
    return m_out
end
