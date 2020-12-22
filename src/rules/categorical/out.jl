export rule

@rule Categorical(:out, Marginalisation) (q_p::Any, ) = begin
    rho = clamp.(exp.(logmean(q_p)), tiny, Inf) # Softens the parameter
    return Categorical(rho ./ sum(rho))
end