using Tullio

@marginalrule DiscreteTransition(:out_in) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    eloga = softmax!(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio result[a, b] := eloga[a, b] * probvec(m_out)[a] * probvec(m_in)[b]
    normalize!(result, 1)
    return Contingency(result, Val(false))
end

@marginalrule DiscreteTransition(:out_in) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, q_a::DirichletCollection, q_T1::PointMass, meta::Any) = begin
    eloga = mean(Base.Broadcast.BroadcastFunction(clamplog), q_a)
    result = eloga[:, :, findfirst(isone, probvec(q_T1))]
    softmax!(result)
    @tullio result[a, b] = result[a, b] * probvec(m_out)[a] * probvec(m_in)[b]
    normalize!(result, 1)
    return Contingency(result, Val(false))
end

@marginalrule DiscreteTransition(:out_in_T1) (m_out::Categorical, m_in::Categorical, m_T1::Categorical, q_a::DirichletCollection, meta::Any) = begin
    eloga = softmax!(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio result[a, b, c] := eloga[a, b, c] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c]
    normalize!(result, 1)
    return Contingency(result, Val(false))
end
