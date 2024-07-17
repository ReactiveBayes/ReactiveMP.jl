@rule TransitionMixture{N}(:switch, Marginalisation) (m_out::Categorical, m_in::Categorical, m_matrices::ManyOf{N, PointMass}) where {N} = begin
    a = zeros(N)
    for k in 1:N
        a[k] = probvec(m_out)' * mean(m_matrices[k]) * probvec(m_in)
    end
    return Categorical(a ./ sum(a))
end

@rule TransitionMixture{N}(:switch, Marginalisation) (m_out::Categorical, m_in::Categorical, q_matrices::ManyOf{N, Union{MatrixDirichlet, PointMass}}) where {N} = begin
    a = zeros(N)
    for k in 1:N
        a[k] = probvec(m_out)' * clamp.(exp.(mean(Base.Broadcast.BroadcastFunction(log), q_matrices[k])), tiny, Inf) * probvec(m_in)
    end
    return Categorical(a ./ sum(a))
end
