
add_to_count_discrete_transition(c, ::Union{PointMass, Categorical, Bernoulli}) = c + 1
add_to_count_discrete_transition(c, ::Contingency{T, <:AbstractArray{T, N}}) where {T, N} = c + N

function ReactiveMP.rule(
    fform::Type{<:DiscreteTransition},
    on::Val{:a},
    vconstraint::Marginalisation,
    messages_names::Nothing,
    messages::Nothing,
    marginals_names::Val{m_names},
    marginals::NTuple{M, Union{Marginal{<:PointMass}, Marginal{<:Categorical}, Marginal{<:Contingency}, Marginal{<:Bernoulli}}},
    meta::Any,
    addons::Any,
    ::Any
) where {M, m_names}
    # Special case, if there is only one marginal, we can return the result directly.
    if M === 1
        return DirichletCollection(components(getdata(first(marginals))) .+ 1), addons
    end
    # First, we have to count the number of dimensions that we need for the contingency matrix.
    c = 0
    foreach(marginals) do marginal
        c = add_to_count_discrete_transition(c, getdata(marginal))
    end
    # Then, we can create the contingency matrix.
    result = ones(ntuple(x -> 1, c))
    foreach(zip(m_names, marginals)) do (marginal_name, marginal)
        # For every incoming marginal distribution, we resolve which dimensions in the final contingency tensor it corresponds to, and then use elementwise multiplication with broadcasting to make sure the correct dimensions are multiplied.
        dims, p = discrete_transition_decode_marginal(String(marginal_name), getdata(marginal))
        localdims = ntuple(dim -> get_corresponding_size(dim, dims, p), c)
        v = reshape(p, localdims)
        result = result .* v
    end
    result = Contingency(result)
    return DirichletCollection(components(result) .+ 1), addons
end
