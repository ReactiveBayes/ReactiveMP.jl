export rule

@rule Bernoulli(:out, Marginalisation) (q_p::Beta, ) = begin
    rho_1 = clamp(exp(logmean(q_p)), tiny, huge)
    rho_2 = clamp(exp(mirroredlogmean(q_p)), tiny, huge)
    return Bernoulli(rho_1 / (rho_1 + rho_2))
end

@rule Bernoulli(:out, Marginalisation) (q_p::PointMass, ) = begin
    return Bernoulli(softmax(mean(q_p)))
end