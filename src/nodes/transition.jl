export Transition

struct Transition end

@node Transition Stochastic [ out, in, a ]

@average_energy Transition (q_out::Any, q_in::Any, q_a::MatrixDirichlet) = begin
    return -probvec(q_out)' * logmean(q_a) * probvec(q_in)
end

@average_energy Transition (q_out_in::Contingency, q_a::MatrixDirichlet) = begin
    return -tr(contingency_matrix(q_out_in)' * logmean(q_a))
end
