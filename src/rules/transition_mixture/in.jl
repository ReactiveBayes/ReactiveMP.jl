@rule TransitionMixture{N}(:in, Marginalisation) (m_out::Categorical, m_switch::Categorical, m_matrices::ManyOf{N, PointMass}) where {N} = begin
    πs = probvec(m_switch)
    ndims = length(probvec(m_out))
    a = tiny * ones(ndims)
    for i in 1:N
        a += πs[i] * mean(m_matrices[i])' * probvec(m_out)
    end
    return Categorical(a ./ sum(a))
end

@rule TransitionMixture{N}(:in, Marginalisation) (m_out::Categorical, m_switch::Categorical, q_matrices::ManyOf{N, Union{MatrixDirichlet, PointMass}}) where {N} = begin
    πs = probvec(m_switch)
    ndims = length(probvec(m_out))
    a = tiny * ones(ndims)
    for i in 1:N
        a += πs[i] * clamp.(exp.(mean(Base.Broadcast.BroadcastFunction(log), q_matrices[i])), tiny, Inf)' * probvec(m_out)
    end
    return Categorical(a ./ sum(a))
end
