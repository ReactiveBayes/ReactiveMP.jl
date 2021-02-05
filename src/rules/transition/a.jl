
@rule Transition(:a, Marginalisation) (q_out::Any, q_in::Categorical) = begin 
    return MatrixDirichlet(probvec(q_out)*probvec(q_in)' .+ 1.0)
end