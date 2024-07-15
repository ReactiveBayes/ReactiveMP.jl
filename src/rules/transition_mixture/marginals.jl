@marginalrule TransitionMixture{N}(:out_in_switch) (
    m_out::Categorical, m_in::Categorical, m_switch::Categorical, q_matrices::ManyOf{N, Union{MatrixDirichlet, PointMass}}
) where {N} = begin
    error(1)
    πs = probvec(m_switch)
    B = Array{Float64, 3}(undef, N, length(πs), length(πs))
    for i in 1:N
        B[i, :, :] = (πs[i] * diagm(probvec(m_out)) * clamp.(exp.(mean(Base.Broadcast.BroadcastFunction(log), q_matrices[i])), tiny, Inf) * diagm(probvec(m_in)))
    end
    return Contingency(B)
end