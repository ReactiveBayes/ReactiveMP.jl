
@rule Transition(:in, Marginalisation) (q_out::Any, q_a::MatrixDirichlet) = begin
    a = clamp.(exp.(logmean(q_a)'*probvec(q_out)), tiny, Inf)
    return Categorical(a ./ sum(a))
end