import Base.Broadcast: BroadcastFunction

@rule DiscreteTransition(:in, Marginalisation) (m_out::Union{DiscreteNonParametric, PointMass}, m_a::PointMass) = begin
    @logscale log(sum(mean(A)' * probvec(m_out)))
    p = mean(m_a)' * probvec(m_out)
    normalize!(p, 1)
    return Categorical(p)
end

@rule DiscreteTransition(:in, Marginalisation) (q_out::Any, q_a::DirichletCollection{T, 2, A}) where {T, A} = begin
    a = clamp.(exp.(mean(BroadcastFunction(log), q_a)' * probvec(q_out)), tiny, Inf)
    return Categorical(a ./ sum(a))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::Union{DiscreteNonParametric, PointMass}, q_a::DirichletCollection{T, 2, A}) where {T, A} = begin
    a = clamp.(exp.(mean(BroadcastFunction(log), q_a))' * probvec(m_out), tiny, Inf)
    return Categorical(a ./ sum(a))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::Union{DiscreteNonParametric, PointMass}, q_a::PointMass, meta::Any) = begin
    return @call_rule DiscreteTransition(:in, Marginalisation) (m_out = m_out, m_a = q_a, meta = meta)
end

@rule DiscreteTransition(:in, Marginalisation) (q_out::PointMass, q_a::PointMass) = begin
    p = mean(q_a)' * mean(q_out)
    normalize!(p, 1)
    return Categorical(p)
end

function ReactiveMP.rule(
    fform::Type{<:DiscreteTransition},
    on::Val{:in},
    vconstraint::Marginalisation,
    messages_names::Val{m_names},
    messages::Tuple,
    marginals_names::Val{(:a,)},
    marginals::Union{Tuple{<:Marginal{<:DirichletCollection}}, Tuple{<:Marginal{<:PointMass}}},
    meta::Any,
    addons::Any,
    ::Any
) where {m_names}
    return __reduce_td_from_messages(messages, first(marginals), 2), addons
end
