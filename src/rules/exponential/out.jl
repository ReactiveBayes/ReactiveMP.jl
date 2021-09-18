export rule

@rule Exponential(:out, Marginalisation) (m_in1::PointMass, ) = begin
    return PointMass(exp(mean(m_in1))
end