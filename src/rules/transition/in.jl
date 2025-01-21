import Base.Broadcast: BroadcastFunction

@rule Transition(:in, Marginalisation) (m_out::Union{DiscreteNonParametric, PointMass}, m_a::PointMass) = begin
    @logscale log(sum(mean(A)' * probvec(m_out)))
    p = mean(m_a)' * probvec(m_out)
    normalize!(p, 1)
    return Categorical(p)
end

@rule Transition(:in, Marginalisation) (q_out::Any, q_a::MatrixDirichlet) = begin
    a = clamp.(exp.(mean(BroadcastFunction(log), q_a)' * probvec(q_out)), tiny, Inf)
    return Categorical(a ./ sum(a))
end

@rule Transition(:in, Marginalisation) (m_out::Union{DiscreteNonParametric, PointMass}, q_a::MatrixDirichlet) = begin
    a = clamp.(exp.(mean(BroadcastFunction(log), q_a))' * probvec(m_out), tiny, Inf)
    return Categorical(a ./ sum(a))
end

@rule Transition(:in, Marginalisation) (m_out::Union{DiscreteNonParametric, PointMass}, q_a::PointMass, meta::Any) = begin
    return @call_rule Transition(:in, Marginalisation) (m_out = m_out, m_a = q_a, meta = meta)
end

@rule Transition(:in, Marginalisation) (q_out::PointMass, q_a::PointMass) = begin
    p = mean(q_a)' * mean(q_out)
    normalize!(p, 1)
    return Categorical(p)
end

function ReactiveMP.rule(
    fform::Type{<:Transition},
    on::Val{:in},
    vconstraint::Marginalisation,
    messages_names::Val{m_names},
    messages::Tuple,
    marginals_names::Val{(:a,)},
    marginals::Tuple,
    meta::Any,
    addons::Any,
    ::Any
) where {m_names}
    # return __reduce_td_from_messages(messages, first(marginals), 2), addons
    q_A = first(marginals)
    e_log_a = mean(BroadcastFunction(log), q_A)
    vmp = clamp.(exp.(e_log_a), tiny, Inf)
    probvecs = probvec.(messages)
    s = size(vmp)
    for i in length(s):-1:3
        vmp = reshape(vmp, (prod(s[1:(i - 1)]), s[i])) * probvecs[i - 1]
    end
    vmp = reshape(vmp, (s[1], s[2]))
    vmp = (first(probvecs)' * vmp)'
    vmp = vmp ./ sum(vmp)
    return Categorical(vmp), addons
end