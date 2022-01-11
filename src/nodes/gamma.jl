
@node Gamma Stochastic [ out, (α, aliases = [ shape ]), (θ, aliases = [ scale ]) ]

@average_energy Gamma (q_out::Any, q_α::PointMass, q_θ::Any) = begin
    mean(loggamma, q_α) + mean(q_α) * mean(log, q_θ) - (mean(q_α) - 1.0) * mean(log, q_out) + mean(q_out)/mean(q_θ)
end

@average_energy Gamma (q_out::Any, q_α::GammaDistributionsFamily, q_θ::Any) = begin
    mean(loggamma, q_α) + mean(q_α) * mean(log, q_θ) - (mean(q_α) - 1.0) * mean(log, q_out) + mean(q_out)/mean(q_θ)
end
