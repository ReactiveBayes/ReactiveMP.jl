export Transition

struct Transition end

@node Transition Stochastic [ out, in, a ]

@average_energy Transition (q_out::Categorical, q_in::Categorical, q_a::MatrixDirichlet) = begin
    return probvec(marg_out)' * logmean(marg_a) * probvec(marg_in1)
end

@average_energy Transition (q_out_in::Any, q_a::MatrixDirichlet) = begin
    return -tr(contingency_matrix(q_out_in)'*logmean(q_a))
end
