using Tullio

@rule DiscreteTransition(:a, Marginalisation) (q_out::PointMass{<:AbstractVector}, q_in::Categorical, meta::Any) = begin
    @tullio result[a, b] := probvec(q_out)[a] * probvec(q_in)[b]
    return DirichletCollection(result .+ 1)
end

@rule DiscreteTransition(:a, Marginalisation) (q_out_in::Contingency, meta::Any) = begin
    return DirichletCollection(components(q_out_in) .+ 1)
end

@rule DiscreteTransition(:a, Marginalisation) (q_out_in::Contingency, q_T1::PointMass{<:AbstractVector{T}}, meta::Any) where {T} = begin
    out_in = components(q_out_in)
    T1 = probvec(q_T1)
    @tullio result[a, b, c] := out_in[a, b] * T1[c]
    return DirichletCollection(result .+ 1)
end
