import Base.Broadcast: BroadcastFunction

__get_corresponding_size(s::NTuple{N, Int}, occurrence::Int) where {N} = s[occurrence]
__get_corresponding_size(s::NTuple{N, Int}, occurrence::Nothing) where {N} = 1

function get_corresponding_size(dim::Int, dims::NTuple{N, Int}, values::AbstractArray{T, N}) where {T, N}
    occurrence = findfirst(==(dim), dims)
    s = size(values)
    return __get_corresponding_size(s, occurrence)
end

"""
    multiply_dimensions(tensor::AbstractArray{T, M}, dims::NTuple{N, Int}, values::AbstractArray{T, N}) where {T, M, N}

Multiply the tensor with the values along the specified dimensions. This is similar to `sum_out_dimensions` but doesn't sum
the result, only performs the elementwise multiplication.
"""
function multiply_dimensions!(tensor::AbstractArray{T, M}, dims::NTuple{N, Int}, values::AbstractArray{T, N}) where {T, M, N}
    localdims = ntuple(dim -> get_corresponding_size(dim, dims, values), M)
    v = reshape(values, localdims)
    tensor .*= v
    return tensor
end

function multiply_dimensions!(tensor::AbstractArray{T, M}, dims::NTuple{N, Int}, values::AbstractArray{P, N}) where {T, M, N, P}
    NT = promote_type(T, P)
    tensor = convert_paramfloattype(NT, tensor)
    values = convert_paramfloattype(NT, values)
    return multiply_dimensions!(tensor, dims, values)
end

"""
    sum_out_dimensions(tensor::AbstractArray{T, M}, dims::NTuple{N, Int}, values::AbstractArray{T, N}) where {T, M, N}

Sum out the dimensions of the tensor that are not part of the marginal distribution. This is a generalization of an inner product, where we also figure out which dimensions of the tensor align with the dimensions of `values`.
"""
function sum_out_dimensions(tensor::AbstractArray{T, M}, dims::NTuple{N, Int}, values::AbstractArray{T, N}) where {T, M, N}
    result = multiply_dimensions!(tensor, dims, values)
    return sum(result, dims = dims)
end

function sum_out_dimensions(tensor::AbstractArray{T, M}, dims::NTuple{N, Int}, values::AbstractArray{P, N}) where {T, M, N, P}
    NT = promote_type(T, P)
    tensor = convert_paramfloattype(NT, tensor)
    values = convert_paramfloattype(NT, values)
    return sum_out_dimensions(tensor, dims, values)
end

function get_corresponding_index(s)
    if s == "out"
        return 1
    elseif s == "in"
        return 2
    else
        return parse(Int, s[2:end]) + 2
    end
end

"""
    discrete_transition_decode_marginal(marginal_name::String, marginal::Contingency{T, <:AbstractArray{T, N}}) where {T, N}

Decode the marginal distribution into a tuple of dimensions and a probability vector.

# Arguments
- `marginal_name`: The name of the marginal distribution.
- `marginal`: The marginal distribution.

For example, if the marginal distribution is "in_t1_t5", we know that "in" corresponds to dimension 2, and "t1" and "t5" correspond to dimensions 3 and 7 of the contingency tensor. 
Therefore, the function will return `(2, 3, 7)` and the contingency tensor `marginal.p`.

"""
function discrete_transition_decode_marginal(marginal_name::String, marginal::Contingency{T, <:AbstractArray{T, N}}) where {T, N}
    split_marginal_names = split(marginal_name, "_")
    dims = Tuple(map(get_corresponding_index, split_marginal_names))::NTuple{N, Int}
    return dims, marginal.p
end

function discrete_transition_decode_marginal(marginal_name::String, marginal::Categorical)
    return (get_corresponding_index(marginal_name),), probvec(marginal)
end

function discrete_transition_decode_marginal(marginal_name::String, marginal::Bernoulli)
    return (get_corresponding_index(marginal_name),), collect(probvec(marginal))
end

function discrete_transition_decode_marginal(marginal_name::String, marginal::PointMass{<:Vector{<:Real}})
    return (get_corresponding_index(marginal_name),), mean(marginal)
end

"""
    discrete_transition_process_marginals(e_log_a, marginals_names, marginals)

Process the marginals to update the expected log transition matrix. This is a common operation used by both
`discrete_transition_structured_message_rule` and `discrete_transition_marginal_rule`.

# Arguments
- `e_log_a`: The expected log of the transition matrix.
- `marginals_names`: The names of the marginal distributions.
- `marginals`: The marginal distributions.

# Returns
- The updated expected log transition matrix.
"""
function discrete_transition_process_marginals(e_log_a, marginals_names, marginals)
    result = copy(e_log_a)
    foreach(zip(marginals_names, marginals)) do (marginal_name, marginal)
        marginal = getdata(marginal)
        if marginal_name === :a
            return nothing
        elseif marginal_name === :out
            result = sum_out_dimensions(result, (1,), probvec(marginal))
        elseif marginal_name === :in
            result = sum_out_dimensions(result, (2,), probvec(marginal))
        else
            dims, p = discrete_transition_decode_marginal(String(marginal_name), marginal)
            result = sum_out_dimensions(result, dims, p)
        end
    end
    return result
end

"""
    discrete_transition_process_messages(e_log_a, message_names, messages, callback)

Process the messages to update the expected log transition matrix. This is a common operation used by both
`discrete_transition_structured_message_rule` and `discrete_transition_marginal_rule`. The `callback` function is used to update the expected log transition matrix. 
This argument toggles between marginalising out a variable (for messages) and computing a joint marginal distribution.
"""
function discrete_transition_process_messages(msg, message_names, messages, callback)
    foreach(zip(message_names, messages)) do (message_name, message)
        if message_name === :out
            msg = callback(msg, (1,), probvec(message))
        elseif message_name === :in
            msg = callback(msg, (2,), probvec(message))
        else
            msg = callback(msg, (get_corresponding_index(String(message_name)),), probvec(message))
        end
    end
    return msg
end

"""
    discrete_transition_structured_message_rule(message_names, messages, marginals_names, marginals, q_a)

Compute the message for one of the Categorical interfaces of the `DiscreteTransition` node. This function 
    1. Computes the expected log of the transition matrix `e_log_a`
    2. For every incoming marginal distribution, it determines which dimension of the contingency tensor it corresponds to.
    3. It then uses `sum_out_dimensions` to to compute the inner product of `e_log_a` with the marginal distribution along the specified dimension.
    4. The result of this is the VMP message, which we have to exponentiate and multiply with the incoming messages.
    5. The result is then normalized to sum to 1.

# Arguments
- `message_names`: The names of the incoming messages. These are the variables in the same factorization cluster as the variable over which we are computing the message.
- `messages`: The incoming messages. These are guaranteed to be either `Categorical`, `Bernoulli` or `PointMass` distributions.
- `marginals_names`: The names of the other marginal distributions attached to the `DiscreteTransition` node. These marginal distributions are not in the same factorization cluster as the variable over which we are computing the message.
- `marginals`: The incoming marginals. These are guaranteed to be either `Contingency`, `Categorical`, `Bernoulli` or `PointMass` distributions.
- `q_a`: The marginal distribution over the transition tensor.
"""
function discrete_transition_structured_message_rule(message_names, messages, marginals_names, marginals, q_a)
    e_log_a = mean(BroadcastFunction(clamplog), q_a)
    e_log_a = discrete_transition_process_marginals(e_log_a, marginals_names, marginals)
    msg = clamp.(exp.(e_log_a), tiny, Inf)
    msg = discrete_transition_process_messages(msg, message_names, messages, sum_out_dimensions)
    msg = reshape(msg, :)
    normalize!(msg, 1)
    return Categorical(msg)
end

function ReactiveMP.rule(
    fform::Type{<:DiscreteTransition},
    on::Val{S},
    vconstraint::Marginalisation,
    messages_names::Val{mes_names},
    messages::NTuple{N, Union{Message{<:PointMass}, Message{<:DiscreteNonParametric}}},
    marginals_names::Val{mar_names},
    marginals::NTuple{M, Union{Marginal{<:DirichletCollection}, Marginal{<:PointMass}, Marginal{<:DiscreteNonParametric}, Marginal{<:Contingency}, Marginal{<:Bernoulli}}},
    meta::Any,
    addons::Any,
    ::Any
) where {S, M, N, mes_names, mar_names}
    q_a = marginals[findfirst(==(:a), mar_names)]
    return discrete_transition_structured_message_rule(mes_names, messages, mar_names, marginals, q_a), addons
end

function ReactiveMP.rule(
    fform::Type{<:DiscreteTransition},
    on::Val{S},
    vconstraint::Marginalisation,
    messages_names::Nothing,
    messages::Nothing,
    marginals_names::Val{mar_names},
    marginals::NTuple{M, Union{Marginal{<:DirichletCollection}, Marginal{<:PointMass}, Marginal{<:DiscreteNonParametric}, Marginal{<:Contingency}, Marginal{<:Bernoulli}}},
    meta::Any,
    addons::Any,
    ::Any
) where {S, M, mar_names}
    q_a = marginals[findfirst(==(:a), mar_names)]
    return discrete_transition_structured_message_rule((), (), mar_names, marginals, q_a), addons
end

# --------------- Custom implementation of some often-used rules ---------------
# These rules are not strictly necessary, but they are more efficient than the generic rule. When using BP, we can use these rules.

using TensorOperations
using Tullio

# --------------- Rules for 2 interfaces (PointMass q_a) ---------------
@rule DiscreteTransition(:out, Marginalisation) (m_in::DiscreteNonParametric, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a)
    @tensor out[i] := eloga[i, a] * probvec(m_in)[a]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::DiscreteNonParametric, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a)
    @tensor out[i] := eloga[a, i] * probvec(m_out)[a]
    return Categorical(normalize!(out, 1))
end

# --------------- Rules for 2 interfaces (q_out PointMass) ---------------

@rule DiscreteTransition(:in, Marginalisation) (q_out::PointMass, q_a::PointMass, meta::Any) = begin
    eloga = mean(Base.Broadcast.BroadcastFunction(clamplog), q_a)
    @tensor out[i] := eloga[a, i] * probvec(q_out)[a]
    out = exp.(out)
    return Categorical(normalize!(out, 1))
end

# --------------- Rules for 2 interfaces (DirichletCollection q_a) ---------------
@rule DiscreteTransition(:out, Marginalisation) (m_in::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    eloga = exp.(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tensor out[i] := eloga[i, a] * probvec(m_in)[a]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    eloga = exp.(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tensor out[i] := eloga[a, i] * probvec(m_out)[a]
    return Categorical(normalize!(out, 1))
end

# --------------- Rules for 2 interfaces (q_out PointMass, q_a DirichletCollection) ---------------

@rule DiscreteTransition(:in, Marginalisation) (q_out::PointMass, q_a::DirichletCollection, meta::Any) = begin
    eloga = mean(Base.Broadcast.BroadcastFunction(clamplog), q_a)
    @tensor out[i] := eloga[a, i] * probvec(q_out)[a]
    out = exp.(out)
    return Categorical(normalize!(out, 1))
end

# --------------- Rules for 3 interfaces (PointMass q_a) ---------------
@rule DiscreteTransition(:out, Marginalisation) (m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a)
    @tensor out[i] := eloga[i, a, b] * probvec(m_in)[a] * probvec(m_T1)[b]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::DiscreteNonParametric, m_T1::DiscreteNonParametric, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a)
    @tensor out[i] := eloga[a, i, b] * probvec(m_out)[a] * probvec(m_T1)[b]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a)
    @tensor out[i] := eloga[a, b, i] * probvec(m_out)[a] * probvec(m_in)[b]
    return Categorical(normalize!(out, 1))
end

# --------------- Rules for 3 interfaces (q_out PointMass, q_a PointMass) ---------------

@rule DiscreteTransition(:in, Marginalisation) (q_out::PointMass, q_a::PointMass, m_T1::DiscreteNonParametric, meta::Any) = begin
    eloga = mean(Base.Broadcast.BroadcastFunction(clamplog), q_a)
    @tensor out[i, j] := eloga[a, i, j] * probvec(q_out)[a]
    out .= exp.(out)
    @tensor msg[i] := out[i, j] * probvec(m_T1)[j]
    return Categorical(normalize!(msg, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (q_out::PointMass, m_in::DiscreteNonParametric, q_a::PointMass, meta::Any) = begin
    eloga = mean(Base.Broadcast.BroadcastFunction(clamplog), q_a)
    @tensor out[i, j] := eloga[a, i, j] * probvec(q_out)[a]
    out .= exp.(out)
    @tensor msg[i] := out[j, i] * probvec(m_in)[j]
    return Categorical(normalize!(msg, 1))
end

# --------------- Rules for 3 interfaces (DirichletCollection q_a) ---------------
@rule DiscreteTransition(:out, Marginalisation) (m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    # @show "rule triggered"
    eloga = exp.(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio out[i] := eloga[i, a, b] * probvec(m_in)[a] * probvec(m_T1)[b]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::DiscreteNonParametric, m_T1::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    eloga = exp.(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tensor out[i] := eloga[a, i, b] * probvec(m_out)[a] * probvec(m_T1)[b]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    eloga = exp.(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tensor out[i] := eloga[a, b, i] * probvec(m_out)[a] * probvec(m_in)[b]
    return Categorical(normalize!(out, 1))
end

# --------------- Rules for 4 interfaces (PointMass q_a) ---------------
@rule DiscreteTransition(:out, Marginalisation) (m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a)
    @tensor out[i] := eloga[i, a, b, c] * probvec(m_in)[a] * probvec(m_T1)[b] * probvec(m_T2)[c]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a)
    @tensor out[i] := eloga[a, i, b, c] * probvec(m_out)[a] * probvec(m_T1)[b] * probvec(m_T2)[c]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T2::DiscreteNonParametric, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a)
    @tensor out[i] := eloga[a, b, i, c] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T2)[c]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, q_a::PointMass, meta::Any) = begin
    eloga = mean(q_a)
    @tensor out[i] := eloga[a, b, c, i] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c]
    return Categorical(normalize!(out, 1))
end

# --------------- Rules for 4 interfaces (DirichletCollection q_a) ---------------
@rule DiscreteTransition(:out, Marginalisation) (m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    eloga = exp.(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tullio out[i] := eloga[i, a, b, c] * probvec(m_in)[a] * probvec(m_T1)[b] * probvec(m_T2)[c]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:in, Marginalisation) (m_out::DiscreteNonParametric, m_T1::DiscreteNonParametric, m_T2::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    eloga = exp.(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tensor out[i] := eloga[a, i, b, c] * probvec(m_out)[a] * probvec(m_T1)[b] * probvec(m_T2)[c]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T1, Marginalisation) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T2::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    eloga = exp.(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tensor out[i] := eloga[a, b, i, c] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T2)[c]
    return Categorical(normalize!(out, 1))
end

@rule DiscreteTransition(:T2, Marginalisation) (m_out::DiscreteNonParametric, m_in::DiscreteNonParametric, m_T1::DiscreteNonParametric, q_a::DirichletCollection, meta::Any) = begin
    eloga = exp.(mean(Base.Broadcast.BroadcastFunction(clamplog), q_a))
    @tensor out[i] := eloga[a, b, c, i] * probvec(m_out)[a] * probvec(m_in)[b] * probvec(m_T1)[c]
    return Categorical(normalize!(out, 1))
end
