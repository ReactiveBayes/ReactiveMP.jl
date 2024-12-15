
@rule Transition(:a, Marginalisation) (q_out::Union{DiscreteNonParametric, PointMass}, q_in::Union{DiscreteNonParametric, PointMass}) = begin
    return MatrixDirichlet(collect(probvec(q_out)) * probvec(q_in)' .+ 1)
end

@rule Transition(:a, Marginalisation) (q_out_in::Contingency,) = begin
    return MatrixDirichlet(components(q_out_in) .+ 1)
end
