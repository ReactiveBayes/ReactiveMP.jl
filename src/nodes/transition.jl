export Transition

struct Transition end

@node Transition Stochastic [out, in, a]

conjugate_type(::Type{<:Transition}, ::Type{Val{:out}}) = Categorical
conjugate_type(::Type{<:Transition}, ::Type{Val{:in}})  = Categorical
conjugate_type(::Type{<:Transition}, ::Type{Val{:a}})   = MatrixDirichlet

@average_energy Transition (q_out::Any, q_in::Any, q_a::MatrixDirichlet) = begin
    return -probvec(q_out)' * mean(log, q_a) * probvec(q_in)
end

@average_energy Transition (q_out_in::Contingency, q_a::MatrixDirichlet) = begin
    return -tr(contingency_matrix(q_out_in)' * mean(log, q_a))
end

@average_energy Transition (q_out_in::Contingency, q_a::PointMass) = begin
    # `map(clamplog, mean(q_a))` is an equivalent of `mean(log, q_a)` with an extra `clamp(el, tiny, Inf)` operation
    # The reason is that we don't want to take log of zeros in the matrix `q_a` (if there are any)
    # The trick here is that if RHS matrix has zero inputs, than the corresponding entries of the `contingency_matrix` matrix 
    # should also be zeros (see corresponding @marginalrule), so at the end `log(tiny) * 0` should not influence the result.
    return -ReactiveMP.mul_trace(ReactiveMP.contingency_matrix(q_out_in)', mean(clamplog, q_a))
end

@average_energy Transition (q_out::Any, q_in::Any, q_a::PointMass) = begin
    return -probvec(q_out)' * mean(clamplog, q_a) * probvec(q_in)
end
