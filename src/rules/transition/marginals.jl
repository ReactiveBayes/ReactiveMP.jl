import Base.Broadcast: BroadcastFunction

@marginalrule Transition(:out_in) (m_out::Categorical, m_in::Categorical, q_a::MatrixDirichlet) = begin
    D = map(e -> clamp(exp(e), tiny, huge), mean(BroadcastFunction(log), q_a))
    B = Diagonal(probvec(m_out)) * D * Diagonal(probvec(m_in))
    P = map!(Base.Fix2(/, sum(B)), B, B) # inplace version of B ./ sum(B)
    return Contingency(P, Val(false))    # Matrix `P` has been normalized by hand
end

@marginalrule Transition(:out_in) (m_out::Categorical, m_in::Categorical, q_a::PointMass) = begin
    B = Diagonal(probvec(m_out)) * mean(q_a) * Diagonal(probvec(m_in))
    P = map!(Base.Fix2(/, sum(B)), B, B) # inplace version of B ./ sum(B)
    return Contingency(P, Val(false))    # Matrix `P` has been normalized by hand
end

@marginalrule Transition(:out_in_a) (m_out::Categorical, m_in::Categorical, m_a::PointMass) = begin
    B = Diagonal(probvec(m_out)) * mean(m_a) * Diagonal(probvec(m_in))
    P = map!(Base.Fix2(/, sum(B)), B, B)                  # inplace version of B ./ sum(B)
    return convert_paramfloattype((out_in = Contingency(P, Val(false)), a = m_a)) # Matrix `P` has been normalized by hand
end

@marginalrule Transition(:out_in_a) (m_out::PointMass, m_in::Categorical, m_a::PointMass, meta::Any) = begin
    m_in_2 = @call_rule Transition(:in, Marginalisation) (m_out = m_out, m_a = m_a, meta = meta)
    return convert_paramfloattype((out = m_out, in = prod(ClosedProd(), m_in_2, m_in), a = m_a))
end

outer_product(vs) = prod.(Iterators.product(vs...))

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