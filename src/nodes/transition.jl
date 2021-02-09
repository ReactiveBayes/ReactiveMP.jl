export Transition

struct Transition end

@node Transition Stochastic [ out, in, a ]

@average_energy Transition (q_out::Categorical, q_in::Categorical, q_a::MatrixDirichlet) = begin
    return probvec(marg_out)' * logmean(marg_a) * probvec(marg_in1)
end