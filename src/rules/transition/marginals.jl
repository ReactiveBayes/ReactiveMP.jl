
@marginalrule Transition(:out_in) (m_out::Categorical, m_in::Categorical, q_a::MatrixDirichlet) = begin
    B = Diagonal(probvec(m_out)) * exp.(mean(log, q_a)) * Diagonal(probvec(m_in))
    return Contingency(B ./ sum(B))
end