@marginalrule TransitionMixture{N}(:out_in_switch) (
    m_out::Categorical, m_in::Categorical, m_switch::Categorical, q_matrices::ManyOf{N, Union{MatrixDirichlet, PointMass}}
) where {N} = begin
    πs = probvec(m_switch)
    π_m_out = probvec(m_out)
    π_m_in = probvec(m_in)
    B = Array{Float64, 3}(undef, N, length(π_m_out), length(π_m_in))
    for i in 1:N
        B[i, :, :] = (πs[i] * diagm(π_m_out) * clamp.(exp.(mean(Base.Broadcast.BroadcastFunction(log), q_matrices[i])), tiny, Inf) * diagm(π_m_in))
    end
    return Contingency(B)
end

function marginalrule(
    ::Type{<:TransitionMixture{N}}, ::Val{:out_in_switch}, ::Val{(:out, :in, :switch)}, messages, qnames::Val{Q}, marginals, meta::Nothing, __node::Nothing
) where {N, Q}
    @assert length(Q) == length(marginals) "The length of `qnames` must match the length of `marginals`"
    @assert N == length(marginals) "The `N` must match the length of `marginals`"
    m_out = getdata(messages[1])
    m_in = getdata(messages[2])
    m_switch = getdata(messages[3])
    q_matrices = getdata.(marginals)
    πs = probvec(m_switch)
    π_m_out = probvec(m_out)
    π_m_in = probvec(m_in)
    B = Array{Float64, 3}(undef, N, length(π_m_out), length(π_m_in))
    for i in 1:N
        B[i, :, :] = (πs[i] * diagm(π_m_out) * clamp.(exp.(mean(Base.Broadcast.BroadcastFunction(log), q_matrices[i])), tiny, Inf) * diagm(π_m_in))
    end
    return Contingency(B)
end
