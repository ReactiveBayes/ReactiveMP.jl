
@node Gamma Stochastic [ out, (α, aliases = [ shape ]), (θ, aliases = [ scale ]) ]

@average_energy Gamma (q_out::Any, q_α::PointMass, q_θ::Any) = begin
    loggammamean(q_α) + mean(q_α) * logmean(q_θ) - (mean(q_α) - 1.0) * logmean(q_out) + mean(q_out)/mean(q_θ)
end

@average_energy Gamma (q_out::Any, q_α::GammaDistributionsFamily, q_θ::Any) = begin
    loggammamean(q_α) + mean(q_α) * logmean(q_θ) - (mean(q_α) - 1.0) * logmean(q_out) + mean(q_out)/mean(q_θ)
end
