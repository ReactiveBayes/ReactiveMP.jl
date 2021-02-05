
@rule Transition(:out, Marginalisation) (q_in::Categorical, q_a::MatrixDirichlet) = begin
    a = clamp.(exp.(logmean(q_a) * probvec(q_in)), tiny, Inf)
    return Categorical(a ./ sum(a))
end