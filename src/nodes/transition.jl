export Transition

struct Transition end

@node Transition Stochastic [ out, in, a ]

conjugate_type(::Type{ <: Transition }, ::Type{ Val{ :out } }) = Categorical
conjugate_type(::Type{ <: Transition }, ::Type{ Val{ :in} })   = Categorical
conjugate_type(::Type{ <: Transition }, ::Type{ Val{ :a } })   = MatrixDirichlet

@average_energy Transition (q_out::Any, q_in::Any, q_a::MatrixDirichlet) = begin
    return -probvec(q_out)' * mean(log, q_a) * probvec(q_in)
end

@average_energy Transition (q_out_in::Contingency, q_a::MatrixDirichlet) = begin
    return -tr(contingency_matrix(q_out_in)' * mean(log, q_a))
end

@average_energy Transition (q_out_in::Contingency, q_a::PointMass) = begin
    return -tr(contingency_matrix(q_out_in)' * mean(log, q_a))
end

@average_energy Transition (q_out::Any, q_in::Any, q_a::PointMass) = begin
    return -probvec(q_out)' * mean(log,q_a) * probvec(q_in)
end