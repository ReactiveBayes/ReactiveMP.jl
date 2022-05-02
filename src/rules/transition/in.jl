
@rule Transition(:in, Marginalisation) (q_out::Any, q_a::MatrixDirichlet) = begin
    a = clamp.(exp.(mean(log, q_a)' * probvec(q_out)), tiny, Inf)
    return Categorical(a ./ sum(a))
end

@rule Transition(:in, Marginalisation) (m_out::Categorical, q_a::MatrixDirichlet) = begin
    a = clamp.(exp.(mean(log, q_a))' * probvec(m_out), tiny, Inf)
    return Categorical(a ./ sum(a))
end

@rule Transition(:in, Marginalisation) (m_out::Any, m_a::PointMass) = begin
    p = mean(m_a)' * probvec(m_out)
    normalize!(p, 1)
    return Categorical(p)
end
