export rule

@rule Categorical(:p, Marginalisation) (q_out::Any, ) = Dirichlet(probvec(q_out) .+ one(eltype(probvec(q_out))))