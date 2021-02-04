
@node Gamma Stochastic [ out, (α, aliases = [ shape ]), (θ, aliases = [ scale ]) ]

@average_energy Gamma (q_out::Any, q_α::Any, q_θ::Any) = begin
    return labsgamma(mean(q_α)) + mean(q_α) * logmean(q_θ) - (mean(q_α) - 1.0) * logmean(q_out) + inv(mean(q_θ)) * mean(q_out)
end
