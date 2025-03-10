import Base.Broadcast: BroadcastFunction

outer_product(vs) = prod.(Iterators.product(vs...))

# Fast implementation for the case where we need a joint marginal over all categoricals.
function marginalrule(
    ::Type{<:DiscreteTransition},
    ::Val{marginal_symbol},
    ::Val{message_names},
    messages::NTuple{N, Union{<:Message{<:Categorical}, <:Message{<:Bernoulli}, <:Message{<:PointMass}}},
    ::Val{(:a)},
    marginals::Tuple{Union{<:Marginal{<:DirichletCollection}}, <:Marginal{<:PointMass}},
    ::Any,
    ::Any
) where {marginal_symbol, message_names, N}
    return Contingency(outer_product(probvec.(messages)) .* clamp.(exp.(mean(BroadcastFunction(log), first(marginals))), tiny, huge))
end

# Generic implementation
"""
    discrete_transition_marginal_rule(message_names, messages, marginals_names, marginals, q_a)

Compute the marginal for one of the Categorical interfaces of the `DiscreteTransition` node. This function is similar to
`discrete_transition_structured_message_rule` but uses `multiply_dimensions` instead of `sum_out_dimensions` for the messages.

# Arguments
- `message_names`: The names of the incoming messages. These are the variables in the same factorization cluster as the variable over which we are computing the message.
- `messages`: The incoming messages. These are guaranteed to be either `Categorical`, `Bernoulli` or `PointMass` distributions.
- `marginals_names`: The names of the other marginal distributions attached to the `DiscreteTransition` node. These marginal distributions are not in the same factorization cluster as the variable over which we are computing the message.
- `marginals`: The incoming marginals. These are guaranteed to be either `Contingency`, `Categorical`, `Bernoulli` or `PointMass` distributions.
- `q_a`: The marginal distribution over the transition tensor.
"""
function discrete_transition_marginal_rule(message_names::NTuple{N, Symbol}, messages, marginals_names::NTuple{M, Symbol}, marginals, q_a) where {N, M}
    e_log_a = mean(BroadcastFunction(log), q_a)
    e_log_a = discrete_transition_process_marginals(e_log_a, marginals_names, marginals)

    marginal = clamp.(exp.(e_log_a), tiny, Inf)
    marginal = discrete_transition_process_messages(marginal, message_names, messages, multiply_dimensions!)
    dims = Tuple(findall(size(marginal) .== 1))::NTuple{M - 1, Int}
    marginal = dropdims(marginal, dims = dims)
    return Contingency(marginal)
end

function marginalrule(
    ::Type{<:DiscreteTransition},
    ::Val{marginal_symbol},
    ::Val{message_names},
    messages::NTuple{N, Union{<:Message{<:Categorical}, <:Message{<:Bernoulli}, <:Message{<:PointMass}}},
    ::Val{marginal_names},
    marginals::NTuple{M, Union{Marginal{<:DirichletCollection}, Marginal{<:PointMass}, Marginal{<:Categorical}, Marginal{<:Contingency}, Marginal{<:Bernoulli}}},
    ::Any,
    ::Any
) where {marginal_symbol, message_names, marginal_names, N, M}
    q_a = marginals[findfirst(==(:a), marginal_names)]
    return discrete_transition_marginal_rule(message_names, messages, marginal_names, marginals, q_a)
end
