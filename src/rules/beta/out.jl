export rule

@rule Beta(:out, Marginalisation) (m_a::PointMass, m_b::PointMass) = begin
    @scaling 0
    Beta(mean(m_a), mean(m_b))
end
@rule Beta(:out, Marginalisation) (q_a::PointMass, q_b::PointMass) = begin
    @scaling 0
    Beta(mean(q_a), mean(q_b))
end
