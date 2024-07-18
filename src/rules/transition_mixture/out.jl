@rule TransitionMixture{N}(:out, Marginalisation) (m_in::Union{Categorical, PointMass}, m_switch::Union{Categorical, PointMass}, m_matrices::ManyOf{N, PointMass}) where {N} = begin
    πs = probvec(m_switch)
    ndims = length(probvec(m_in))
    a = tiny * ones(ndims)
    for i in 1:N
        a += πs[i] * mean(m_matrices[i]) * probvec(m_in)
    end
    return Categorical(a ./ sum(a))
end

@rule TransitionMixture{N}(:out, Marginalisation) (
    m_in::Union{Categorical, PointMass}, m_switch::Union{Categorical, PointMass}, q_matrices::ManyOf{N, Union{MatrixDirichlet, PointMass}}
) where {N} = begin
    πs = probvec(m_switch)
    ndims = length(probvec(m_in))
    a = tiny * ones(ndims)
    for i in 1:N
        a += πs[i] * clamp.(exp.(mean(Base.Broadcast.BroadcastFunction(log), q_matrices[i])), tiny, Inf) * probvec(m_in)
    end
    return Categorical(a ./ sum(a))
end
