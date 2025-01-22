export Transition

import Base.Broadcast: BroadcastFunction

struct Transition end

ReactiveMP.sdtype(::Type{Transition}) = ReactiveMP.Stochastic()
ReactiveMP.is_predefined_node(::Type{Transition}) = ReactiveMP.PredefinedNodeFunctionalForm()

function ReactiveMP.prepare_interfaces_generic(fform::Type{Transition}, interfaces::AbstractVector)
    return map(enumerate(interfaces)) do (index, (name, variable))
        return ReactiveMP.NodeInterface(ReactiveMP.alias_interface(fform, index, name), variable)
    end
end

function ReactiveMP.alias_interface(::Type{Transition}, index, name)
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

function ReactiveMP.collect_factorisation(::Type{Transition}, t::Tuple)
    return t
end

@average_energy Transition (q_out::Any, q_in::Any, q_a::MatrixDirichlet) = begin
    return -probvec(q_out)' * mean(BroadcastFunction(log), q_a) * probvec(q_in)
end

@average_energy Transition (q_out_in::Contingency, q_a::MatrixDirichlet) = begin
    return -tr(components(q_out_in)' * mean(BroadcastFunction(log), q_a))
end

@average_energy Transition (q_out_in::Contingency, q_a::PointMass) = begin
    # `map(clamplog, mean(q_a))` is an equivalent of `mean(BroadcastFunction(log), q_a)` with an extra `clamp(el, tiny, Inf)` operation
    # The reason is that we don't want to take log of zeros in the matrix `q_a` (if there are any)
    # The trick here is that if RHS matrix has zero inputs, than the corresponding entries of the `contingency_matrix` matrix 
    # should also be zeros (see corresponding @marginalrule), so at the end `log(tiny) * 0` should not influence the result.
    result = -ReactiveMP.mul_trace(components(q_out_in)', mean(BroadcastFunction(clamplog), q_a))
    return result
end

@average_energy Transition (q_out::Any, q_in::Any, q_a::PointMass) = begin
    return -probvec(q_out)' * mean(BroadcastFunction(clamplog), q_a) * probvec(q_in)
end

function score(::AverageEnergy, ::Type{<:Transition}, ::Val{mnames}, marginals::Tuple{<:Marginal{<:Contingency}, <:Marginal{<:TensorDirichlet}}, ::Nothing) where {mnames}
    q_contingency, q_a = getdata.(marginals)
    return -sum(mean(BroadcastFunction(log), q_a) .* components(q_contingency))
end

function __reduce_td_from_messages(messages, q_A, interface_index)
    vmp = clamp.(exp.(mean(BroadcastFunction(log), q_A)), tiny, Inf)
    probvecs = probvec.(messages)
    for (i, vector) in enumerate(probvecs)
        if i ≥ interface_index
            actual_index = i + 1
        else
            actual_index = i
        end
        v = view(vector, :)
        localdims = ntuple(x -> x == actual_index::Int64 ? length(v) : 1, ndims(vmp))
        vmp .*= reshape(v, localdims)
    end
    dims = ntuple(x -> x ≥ interface_index ? x + 1 : x, ndims(vmp) - 1)
    vmp = sum(vmp, dims = dims)
    msg = reshape(vmp, :)
    msg ./= sum(msg)
    return Categorical(msg)
end
