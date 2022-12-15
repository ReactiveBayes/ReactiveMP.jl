
@rule Beta(:out, Marginalisation) (m_a::PointMass, m_b::PointMass) = begin
    @logscale 0
    return Beta(mean(m_a), mean(m_b))
end

@rule Beta(:out, Marginalisation) (q_a::PointMass, q_b::PointMass) = begin
    @logscale 0
    return Beta(mean(q_a), mean(q_b))
end
