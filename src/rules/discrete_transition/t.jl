import Base.Broadcast: BroadcastFunction

function ReactiveMP.rule(
    fform::Type{<:DiscreteTransition},
    on::Val{S},
    vconstraint::Marginalisation,
    messages_names::Val{m_names},
    messages::Tuple,
    marginals_names::Val{(:a,)},
    marginals::Union{Tuple{<:Marginal{<:DirichletCollection}}, Tuple{<:Marginal{<:PointMass}}},
    meta::Any,
    addons::Any,
    ::Any
) where {S, m_names}
    interface_index = parse(Int, String(S)[2:end]) + 2
    return __reduce_td_from_messages(messages, first(marginals), interface_index), addons
end
