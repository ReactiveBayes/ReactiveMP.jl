using Tullio

@marginalrule DiscreteTransition(:out_in) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, q_a::DirichletCollection) = begin
    eloga = exp.(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio result[a, b] := eloga[a, b] * probvec(m_out)[a] * probvec(m_in)[b]
    return Contingency(result)
end

@marginalrule DiscreteTransition(:out_in) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, q_a::DirichletCollection, q_T1::PointMass) = begin
    eloga = mean(Base.Broadcast.BroadcastFunction(clamplog), q_a)
    @tullio result[a, b] := eloga[a, b, i] * probvec(q_T1)[i]
    result = exp.(result)
    @tullio result[a, b] = result[a, b] * probvec(m_out)[a] * probvec(m_in)[b]
    return Contingency(result)
end

@marginalrule DiscreteTransition(:out_in_T1) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, q_a::DirichletCollection) = begin
    eloga = exp.(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio result[a, b, c] := eloga[a, b, c] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c]
    return Contingency(result)
end