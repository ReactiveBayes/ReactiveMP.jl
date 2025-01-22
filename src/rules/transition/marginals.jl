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

@marginalrule Transition(:out_in) (m_out::PointMass, m_in::Categorical, q_a::PointMass) = begin
    m_in_2 = @call_rule Transition(:in, Marginalisation) (m_out = m_out, q_a = q_a)
    return convert_paramfloattype((out = m_out, in = prod(ClosedProd(), m_in, m_in_2)))
end

outer_product(vs) = prod.(Iterators.product(vs...))

function marginalrule(
    ::Type{<:Transition}, ::Val{marginal_symbol}, ::Val{message_names}, messages::Tuple, ::Val{marginal_names}, marginals::Tuple, ::Any, ::Any
) where {marginal_symbol, message_names, marginal_names}
    return Contingency(outer_product(probvec.(messages)) .* clamp.(exp.(mean(BroadcastFunction(log), first(marginals))), tiny, huge))
end
