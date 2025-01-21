import Base.Broadcast: BroadcastFunction

# Belief Propagation                #
# --------------------------------- #

@rule Transition(:out, Marginalisation) (m_in::Union{PointMass, DiscreteNonParametric}, m_a::PointMass) = begin
    @logscale 0
    p = mean(m_a) * probvec(m_in)
    normalize!(p, 1)
    return Categorical(p)
end

@rule Transition(:out, Marginalisation) (q_in::PointMass, q_a::PointMass) = begin
    @logscale 0
    p = mean(q_a) * mean(q_in)
    normalize!(p, 1)
    return Categorical(p)
end

# Variational                       # 
# --------------------------------- #

@rule Transition(:out, Marginalisation) (q_in::DiscreteNonParametric, q_a::Any) = begin
    a = clamp.(exp.(mean(BroadcastFunction(log), q_a) * probvec(q_in)), tiny, Inf)
    return Categorical(a ./ sum(a))
end

@rule Transition(:out, Marginalisation) (m_in::DiscreteNonParametric, q_a::ContinuousMatrixDistribution) = begin
    a = clamp.(exp.(mean(BroadcastFunction(log), q_a)) * probvec(m_in), tiny, Inf)
    return Categorical(a ./ sum(a))
end

@rule Transition(:out, Marginalisation) (m_in::Union{PointMass, DiscreteNonParametric}, q_a::PointMass, meta::Any) = begin
    @logscale 0
    return @call_rule Transition(:out, Marginalisation) (m_in = m_in, m_a = q_a, meta = meta, addons = getaddons())
end

function ReactiveMP.rule(
    fform::Type{<:Transition},
    on::Val{:out},
    vconstraint::Marginalisation,
    messages_names::Val{m_names},
    messages::Tuple,
    marginals_names::Val{(:a,)},
    marginals::Tuple,
    meta::Any,
    addons::Any,
    ::Any
) where {m_names}
    # return __reduce_td_from_messages(messages, first(marginals), 1), addons

    # Keep this because this is faster.
    q_A = first(marginals)
    e_log_a = mean(Base.Broadcast.BroadcastFunction(log), q_A)
    vmp = clamp.(exp.(e_log_a), tiny, Inf)
    probvecs = probvec.(messages)
    s = size(vmp)
    for i in length(s):-1:2
        vmp = reshape(vmp, (prod(s[1:(i - 1)]), s[i])) * probvecs[i - 1]
    end
    vmp = vmp ./ sum(vmp)
    return Categorical(vmp), addons
end