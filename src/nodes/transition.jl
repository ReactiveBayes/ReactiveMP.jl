export Transition

import Base.Broadcast: BroadcastFunction

struct Transition end

@node Transition Stochastic [out, in, a]

@average_energy Transition (q_out::Any, q_in::Any, q_a::MatrixDirichlet) = begin
    return -probvec(q_out)' * mean(BroadcastFunction(log), q_a) * probvec(q_in)
end

@average_energy Transition (q_out_in::Contingency, q_a::MatrixDirichlet) = begin
    return -tr(components(q_out_in)' * mean(BroadcastFunction(log), q_a))
end

@average_energy Transition (q_out_in::Contingency, q_a::PointMass) = begin
    # `map(clamplog, mean(q_a))` is an equivalent of `mean(log, q_a)` with an extra `clamp(el, tiny, Inf)` operation
    # The reason is that we don't want to take log of zeros in the matrix `q_a` (if there are any)
    # The trick here is that if RHS matrix has zero inputs, than the corresponding entries of the `contingency_matrix` matrix 
    # should also be zeros (see corresponding @marginalrule), so at the end `log(tiny) * 0` should not influence the result.
    return -ReactiveMP.mul_trace(components(q_out_in)', mean(BroadcastFunction(clamplog), q_a))
end

@average_energy Transition (q_out::Any, q_in::Any, q_a::PointMass) = begin
    return -probvec(q_out)' * mean(BroadcastFunction(clamplog), q_a) * probvec(q_in)
end
