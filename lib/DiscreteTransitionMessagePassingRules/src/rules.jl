# The rules, each one contraction over whatever inputs the factorisation delivers (`default`),
# as v6's generic ones. The inputs that are marginals sum their axes out of `E[log A]` before it
# is exponentiated, and the messages multiply the result along theirs.

const BroadcastFunction = Base.Broadcast.BroadcastFunction

# `E[log A]`, clamped away from `log 0`. An integer tensor takes the float type of the other
# inputs, `inputs` being `key => value` pairs, as v6's rules took theirs from the messages.
expected_log_tensor(q_a, inputs) = mean(BroadcastFunction(clamplog), q_a)
expected_log_tensor(q_a::PointMass{<:AbstractArray{<:Integer}}, inputs) = clamplog.(float_tensor(q_a, inputs))

input_float_type(inputs) = float(mapreduce(((key, d),) -> key === :a ? Bool : eltype(discrete_transition_weights(d)), promote_type, inputs; init = Bool))
float_tensor(q_a::PointMass{<:AbstractArray{<:Integer}}, inputs) = input_float_type(inputs).(mean(q_a))
float_tensor(q_a::PointMass, inputs) = mean(q_a)

# `exp(E[log A])` with the marginals summed out of `E[log A]` first, up to a constant factor.
# A point mass with nothing to sum out is `A` itself, clamped as `clamplog` would: no logarithm
# and no exponential, as v6's explicit rules for belief propagation computed it.
function exponentiated_tensor(q_a, marginals, inputs)
    if q_a isa PointMass && all(((key, _),) -> key === :a, marginals)
        return clamp.(float_tensor(q_a, inputs), tiny, huge)
    end
    # A fresh tensor, which the caller owns.
    tensor = contract_marginals(expected_log_tensor(q_a, inputs), marginals)
    return clamp!(softmax!(tensor), tiny, huge)
end

# `E[log A]` with every marginal but `q(a)` summed out along its axes.
function contract_marginals(tensor, marginals; skip = ())
    for (key, marginal) in marginals
        (key === :a || key in skip) && continue
        w = discrete_transition_weights(marginal)
        tensor = sum_out_dimensions(tensor, discrete_transition_axes(key, ndims(w)), w)
    end
    return tensor
end

# The message towards a categorical interface: the target's axis is the one left.
function discrete_transition_message(args)
    messages, marginals = rule_inputs(DiscreteTransition, args.m), rule_inputs(DiscreteTransition, args.q)
    message = exponentiated_tensor(args.q[:a], marginals, (messages..., marginals...))
    for (key, m) in messages
        w = discrete_transition_weights(m)
        message = sum_out_dimensions(message, discrete_transition_axes(key, ndims(w)), w)
    end
    return Categorical(normalize!(reshape(message, :), 1); check_args = false)
end

@define_message_update_rule(
    node = DiscreteTransition, target = :out, args = (default, q[:a]::DiscreteTransitionTensor),
    body = (args) -> discrete_transition_message(args),
)

@define_message_update_rule(
    node = DiscreteTransition, target = :in, args = (default, q[:a]::DiscreteTransitionTensor),
    body = (args) -> discrete_transition_message(args),
)

@define_message_update_rule(
    node = DiscreteTransition, target = (:T, k), args = (default, q[:a]::DiscreteTransitionTensor),
    body = (args) -> discrete_transition_message(args),
)

# Towards `a`: the expected counts, the outer product of the marginals along their axes, plus one.
@define_message_update_rule(
    node = DiscreteTransition, target = :a, args = (default,),
    body = (args) -> begin
        inputs = rule_inputs(DiscreteTransition, args.q)
        placed = map(((key, q),) -> (w = discrete_transition_weights(q); (discrete_transition_axes(key, ndims(w)), w)), inputs)
        n = maximum(((axes, _),) -> maximum(axes), placed)
        counts = ones(promote_type(map(((_, w),) -> eltype(w), placed)...), ntuple(_ -> 1, n))
        for (axes, w) in placed
            counts = counts .* reshape(w, ntuple(dim -> corresponding_size(dim, axes, w), n))
        end
        DirichletCollection(counts .+ 1)
    end,
)

# The marginal of any cluster: the messages of its members multiply the exponentiated tensor
# along their axes. Point-mass messages are observations: they are summed out like marginals,
# and those that are members of the cluster in their own right are split off as blocks of their
# own, as v6 did. One inside a whole group `T` stays in the joint, as a one-hot axis.
@define_marginal_update_rule(
    node = DiscreteTransition, target = members, args = (default, q[:a]::DiscreteTransitionTensor),
    body = (args) -> begin
        messages = rule_inputs(DiscreteTransition, args.m)
        split = filter(((key, m),) -> m isa PointMass && key in members, messages)
        joined = filter(((key, m),) -> !(m isa PointMass && key in members), messages)
        marginals = rule_inputs(DiscreteTransition, args.q)
        joint = exponentiated_tensor(args.q[:a], (marginals..., split...), (messages..., marginals...))
        for (key, m) in joined
            w = discrete_transition_weights(m)
            joint = multiply_dimensions!(joint, discrete_transition_axes(key, ndims(w)), w)
        end
        joint = dropdims(joint; dims = Tuple(findall(==(1), size(joint))))
        normalize!(joint, 1)
        distribution = ndims(joint) == 1 ? Categorical(joint; check_args = false) : Contingency(joint, Val(false))
        isempty(split) && return distribution
        # The split members' point masses and the joint of the rest, as blocks in cluster order.
        observed = map(first, split)
        rest = filter(member -> !(member in observed), members)
        blocks = Any[(key,) => m for (key, m) in split]
        isempty(rest) || push!(blocks, rest => distribution)
        sort!(blocks; by = block -> findfirst(==(first(first(block))), members))
        FactorizedCluster(blocks...)
    end,
)

# ⟨-log A[out, in, T…]⟩: `E[log A]` weighted by every marginal along its axes, summed.
@define_average_energy(
    node = DiscreteTransition, args = (default, q[:a]::DiscreteTransitionTensor),
    body = (args) -> begin
        marginals = rule_inputs(DiscreteTransition, args.q)
        tensor = copy(expected_log_tensor(args.q[:a], marginals))
        for (key, q) in marginals
            key === :a && continue
            w = discrete_transition_weights(q)
            tensor = multiply_dimensions!(tensor, discrete_transition_axes(key, ndims(w)), w)
        end
        -sum(tensor)
    end,
)
