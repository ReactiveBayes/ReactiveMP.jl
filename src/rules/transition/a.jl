
@rule Transition(:a, Marginalisation) (q_out::Any, q_in::Categorical) = begin
    return MatrixDirichlet(collect(probvec(q_out)) * probvec(q_in)' .+ 1)
end

@rule Transition(:a, Marginalisation) (q_out_in::Contingency,) = begin
    return MatrixDirichlet(components(q_out_in) .+ 1)
end
