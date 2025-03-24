export DiscreteTransition

import Base.Broadcast: BroadcastFunction

struct DiscreteTransition end

ReactiveMP.sdtype(::Type{DiscreteTransition}) = ReactiveMP.Stochastic()
ReactiveMP.is_predefined_node(::Type{DiscreteTransition}) = ReactiveMP.PredefinedNodeFunctionalForm()

function ReactiveMP.prepare_interfaces_generic(fform::Type{DiscreteTransition}, interfaces::AbstractVector)
    return map(enumerate(interfaces)) do (index, (name, variable))
        return ReactiveMP.NodeInterface(ReactiveMP.alias_interface(fform, index, name), variable)
    end
end

function ReactiveMP.alias_interface(::Type{DiscreteTransition}, index, name)
    if name === :out && index === 1
        return :out
    elseif name === :in && index === 2
        return :in
    elseif name === :in && index === 3
        return :a
    elseif name === :in && index >= 4
        return Symbol(:T, index - 3)
    end
end

function ReactiveMP.collect_factorisation(::Type{DiscreteTransition}, t::Tuple)
    return t
end

@average_energy DiscreteTransition (q_out::Any, q_in::Any, q_a::DirichletCollection{T, 2, A}) where {T, A} = begin
    return -probvec(q_out)' * mean(BroadcastFunction(log), q_a) * probvec(q_in)
end

@average_energy DiscreteTransition (q_out_in::Contingency, q_a::DirichletCollection{T, 2, A}) where {T, A} = begin
    return -tr(components(q_out_in)' * mean(BroadcastFunction(log), q_a))
end

@average_energy DiscreteTransition (q_out_in::Contingency, q_a::PointMass) = begin
    # `map(clamplog, mean(q_a))` is an equivalent of `mean(BroadcastFunction(log), q_a)` with an extra `clamp(el, tiny, Inf)` operation
    # The reason is that we don't want to take log of zeros in the matrix `q_a` (if there are any)
    # The trick here is that if RHS matrix has zero inputs, than the corresponding entries of the `contingency_matrix` matrix 
    # should also be zeros (see corresponding @marginalrule), so at the end `log(tiny) * 0` should not influence the result.
    result = -ReactiveMP.mul_trace(components(q_out_in)', mean(BroadcastFunction(clamplog), q_a))
    return result
end

@average_energy DiscreteTransition (q_out::Any, q_in::Any, q_a::PointMass) = begin
    return -probvec(q_out)' * mean(BroadcastFunction(clamplog), q_a) * probvec(q_in)
end

function score(::AverageEnergy, ::Type{<:DiscreteTransition}, ::Val{mnames}, marginals::Tuple{<:Marginal{<:Contingency}, <:Marginal{<:DirichletCollection}}, ::Any) where {mnames}
    q_contingency, q_a = getdata.(marginals)
    return -sum(mean(BroadcastFunction(clamplog), q_a) .* components(q_contingency))
end

function score(
    ::AverageEnergy,
    ::Type{<:DiscreteTransition},
    ::Val{mnames},
    marginals::NTuple{N, Union{<:Marginal{Bernoulli}, <:Marginal{Categorical}, <:Marginal{<:Contingency}, <:Marginal{<:DirichletCollection}, <:Marginal{<:PointMass}}},
    ::Any
) where {mnames, N}
    q_a = marginals[findfirst(==(:a), mnames)]
    e_log_a = mean(BroadcastFunction(clamplog), q_a)
    foreach(zip(mnames, marginals)) do (marginal_name, marginal)
        marginal = getdata(marginal)
        if marginal_name === :a
            return nothing
        elseif marginal_name === :out
            e_log_a = multiply_dimensions!(e_log_a, (1,), probvec(marginal))
        elseif marginal_name === :in
            e_log_a = multiply_dimensions!(e_log_a, (2,), probvec(marginal))
        else
            dims, p = discrete_transition_decode_marginal(String(marginal_name), marginal)
            e_log_a = multiply_dimensions!(e_log_a, dims, p)
        end
    end
    return -sum(e_log_a)
end
