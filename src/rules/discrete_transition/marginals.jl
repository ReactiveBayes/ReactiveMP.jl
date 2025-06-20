import Base.Broadcast: BroadcastFunction

outer_product(vs) = prod.(Iterators.product(vs...))

# Fast implementation for the case where we need a joint marginal over all categoricals.
function marginalrule(
    ::Type{<:DiscreteTransition},
    ::Val{marginal_symbol},
    ::Val{message_names},
    messages::NTuple{N, Union{<:Message{<:DiscreteNonParametric}, <:Message{<:Bernoulli}}},
    ::Val{(:a)},
    marginals::Tuple{Union{<:Marginal{<:DirichletCollection}}, <:Marginal{<:PointMass}},
    ::Any,
    ::Any
) where {marginal_symbol, message_names, N}
    result = outer_product(probvec.(messages)) .* exp.(mean(BroadcastFunction(clamplog), first(marginals)))
    normalize!(result, 1)
    return Contingency(result, Val(false))
end

nonparametric_distribution(v::Vector{<:Real}) = Categorical(normalize!(v, 1); check_args = false)
nonparametric_distribution(v::AbstractArray{<:Real, N} where {N}) = Contingency(normalize!(v, 1), Val(false))

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
function discrete_transition_marginal_rule(
    message_names::NTuple{N, Symbol}, messages::NTuple{N, Union{<:Message{<:DiscreteNonParametric}, <:Message{<:Bernoulli}}}, marginals_names::NTuple{M, Symbol}, marginals, q_a
) where {N, M}
    e_log_a = mean(BroadcastFunction(clamplog), q_a)
    e_log_a = discrete_transition_process_marginals(e_log_a, marginals_names, marginals)

    marginal = clamp.(exp.(e_log_a), tiny, huge)
    marginal = discrete_transition_process_messages(marginal, message_names, messages, multiply_dimensions!)
    dims = Tuple(findall(size(marginal) .== 1))
    marginal = dropdims(marginal, dims = dims)
    normalize!(marginal, 1)
    return marginal
end

discrete_transition_marginal_rule_contingency(
    message_names::NTuple{N, Symbol}, messages::NTuple{N, Union{<:Message{<:DiscreteNonParametric}, <:Message{<:Bernoulli}}}, marginals_names::NTuple{M, Symbol}, marginals, q_a
) where {N, M} = Contingency(discrete_transition_marginal_rule(message_names, messages, marginals_names, marginals, q_a), Val(false))

function marginalrule(
    ::Type{<:DiscreteTransition},
    ::Val{marginal_symbol},
    ::Val{message_names},
    messages::NTuple{N, Union{<:Message{<:DiscreteNonParametric}, <:Message{<:Bernoulli}}},
    ::Val{marginal_names},
    marginals::NTuple{M, Union{Marginal{<:DirichletCollection}, Marginal{<:PointMass}, Marginal{<:Categorical}, Marginal{<:Contingency}, Marginal{<:Bernoulli}}},
    ::Any,
    ::Any
) where {marginal_symbol, message_names, marginal_names, N, M}
    q_a = marginals[findfirst(==(:a), marginal_names)]
    return discrete_transition_marginal_rule_contingency(message_names, messages, marginal_names, marginals, q_a)
end

function marginalrule(
    ::Type{<:DiscreteTransition},
    ::Val{marginal_symbol},
    ::Val{message_names},
    messages::NTuple{N, Union{<:Message{<:DiscreteNonParametric}, <:Message{<:Bernoulli}, <:Message{<:PointMass}}},
    ::Val{marginal_names},
    marginals::NTuple{M, Union{Marginal{<:DirichletCollection}, Marginal{<:PointMass}, Marginal{<:Categorical}, Marginal{<:Contingency}, Marginal{<:Bernoulli}}},
    ::Any,
    ::Any
) where {marginal_symbol, message_names, marginal_names, N, M}
    # Find indices of PointMass and non-PointMass messages
    point_mass_indices = findall(m -> m isa Message{<:PointMass}, messages)
    remaining_indices = setdiff(1:length(messages), point_mass_indices)

    # Create NamedTuple for PointMass messages
    point_mass_names = message_names[point_mass_indices]
    msg_point_mass_tuple = NamedTuple{Tuple(point_mass_names)}(Tuple(messages[i] for i in point_mass_indices))
    point_mass_tuple = NamedTuple{Tuple(point_mass_names)}(Tuple(getdata(messages[i]) for i in point_mass_indices))

    if isempty(remaining_indices)
        # If all messages are PointMass, return just the PointMass tuple
        return point_mass_tuple
    else
        # Process remaining non-PointMass messages
        remaining_names = message_names[remaining_indices]
        remaining_messages = Tuple(messages[i] for i in remaining_indices)

        # Get transition tensor marginal
        q_a = marginals[findfirst(==(:a), marginal_names)]

        n_marginal_names = (marginal_names..., point_mass_names...)
        n_marginals = (marginals..., msg_point_mass_tuple...)
        # Compute joint marginal for non-PointMass messages
        joint_marginal = nonparametric_distribution(discrete_transition_marginal_rule(remaining_names, remaining_messages, n_marginal_names, n_marginals, q_a))

        # Create name for joint distribution by concatenating remaining names
        joint_name = Symbol(join(remaining_names, '_'))
        joint_tuple = NamedTuple{(joint_name,)}((joint_marginal,))

        resulting_names = []
        resulting_collection = []
        for i in 1:length(messages)
            if i in point_mass_indices
                push!(resulting_collection, getdata(messages[i]))
                push!(resulting_names, message_names[i])
            elseif i == first(remaining_indices)
                push!(resulting_collection, joint_marginal)
                push!(resulting_names, joint_name)
            end
        end

        # Merge PointMass and joint marginal results in order of occurrence in messages
        result = NamedTuple{Tuple(resulting_names)}(Tuple(resulting_collection))
        return result
    end
end
