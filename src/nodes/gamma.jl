
@node Gamma Stochastic [ out, (α, aliases = [ shape ]), (θ, aliases = [ scale ]) ]

# TODO: Removed due to possible mistake
# @average_energy Gamma (q_out::Any, q_α::PointMass, q_θ::Any) = begin
#     loggammamean(q_α) - mean(q_α) * logmean(q_θ) - (mean(q_α) - 1.0) * logmean(q_out) + mean(q_θ) * mean(q_out)
# end

# @average_energy Gamma (q_out::Any, q_α::GammaDistributionsFamily, q_θ::Any) = begin
#     meanlogmean(q_α) - mean(q_α) + 0.5 * (log2π - logmean(q_α)) - mean(q_α) * logmean(q_θ) - (mean(q_α) - 1.0) * logmean(q_out) + mean(q_θ) * mean(q_out)
# end