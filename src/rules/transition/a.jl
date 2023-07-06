
@rule Transition(:a, Marginalisation) (q_out::Any, q_in::Categorical) = begin
    return MatrixDirichlet(probvec(q_out) * probvec(q_in)' .+ 1.0)
end

@rule Transition(:a, Marginalisation) (q_out_in::Contingency,) = begin
    return MatrixDirichlet(contingency_matrix(q_out_in) .+ 1.0)
end

@rule Transition(:a, Marginalisation) (q_out_in::Contingency, q_out::PointMass, ) = begin 
    ## scale factor
    @logscale 0 # incorrect; don't know the exact number  
    ###
    return MatrixDirichlet(contingency_matrix(q_out_in) .+ 1.0)
end