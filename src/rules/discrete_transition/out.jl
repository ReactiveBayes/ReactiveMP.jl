import Base.Broadcast: BroadcastFunction

# Belief Propagation                #
# --------------------------------- #

@rule DiscreteTransition(:out, Marginalisation) (m_in::Union{PointMass{<:Vector{<:Real}}, DiscreteNonParametric}, m_a::PointMass{<:Matrix{<:Real}}) = begin
    @logscale 0
    p = mean(m_a) * probvec(m_in)
    normalize!(p, 1)
    return Categorical(p)
end

@rule DiscreteTransition(:out, Marginalisation) (q_in::PointMass{<:Vector{<:Real}}, q_a::PointMass{<:Matrix{<:Real}}) = begin
    @logscale 0
    p = mean(q_a) * mean(q_in)
    normalize!(p, 1)
    return Categorical(p)
end

# Variational                       # 
# --------------------------------- #

@rule DiscreteTransition(:out, Marginalisation) (q_in::DiscreteNonParametric, q_a::Union{DirichletCollection{T, 2, A} where {T, A}, PointMass{<:Matrix{<:Real}}}) = begin
    a = clamp.(exp.(mean(BroadcastFunction(log), q_a) * probvec(q_in)), tiny, Inf)
    return Categorical(a ./ sum(a))
end

@rule DiscreteTransition(:out, Marginalisation) (m_in::DiscreteNonParametric, q_a::Union{DirichletCollection{T, 2, A} where {T, A}, PointMass{<:Matrix{<:Real}}}) = begin
    a = clamp.(exp.(mean(BroadcastFunction(log), q_a)) * probvec(m_in), tiny, Inf)
    return Categorical(a ./ sum(a))
end

function ReactiveMP.rule(
    fform::Type{<:DiscreteTransition},
    on::Val{:out},
    vconstraint::Marginalisation,
    messages_names::Val{m_names},
    messages::Tuple,
    marginals_names::Val{(:a,)},
    marginals::Union{Tuple{<:Marginal{<:DirichletCollection}}, Tuple{<:Marginal{<:PointMass}}},
    meta::Any,
    addons::Any,
    ::Any
) where {m_names}
    return __reduce_td_from_messages(messages, first(marginals), 1), addons
end
