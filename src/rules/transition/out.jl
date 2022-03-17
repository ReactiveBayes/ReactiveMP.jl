
@rule Transition(:out, Marginalisation) (q_in::Categorical, q_a::MatrixDirichlet) = begin
    a = clamp.(exp.(mean(log, q_a) * probvec(q_in)), tiny, Inf)
    return Categorical(a ./ sum(a))
end

@rule Transition(:out, Marginalisation) (m_in::Categorical, q_a::MatrixDirichlet) = begin
    a = clamp.(exp.(mean(log, q_a)) * probvec(m_in), tiny, Inf)
    return Categorical(a ./ sum(a))
end

@rule Transition(:out, Marginalisation) (m_in::Any, m_a::PointMass) = begin
    p = mean(m_a) * probvec(m_in)
    normalize!(p, 1)
    return Categorical(p)
end