@rule Mixture((:inputs, k), Marginalisation) (m_out::Any, m_switch::Any) = begin
    # `messages` are available from the `@rule` macro itself
    @logscale getlogscale(messages[1]) + getlogscale(messages[2]) + log(probvec(messages[2])[k])
    return m_out
end

@rule Mixture((:inputs, k), Marginalisation) (m_out::Any, q_switch::PointMass) = begin
    @logscale getlogscale(marginals[1])
    return m_out
end

