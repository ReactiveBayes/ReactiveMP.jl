
import StatsFuns: log2π

@node GammaShapeRate Stochastic [ out, (α, aliases = [ shape ]), (β, aliases = [ rate ]) ]

@average_energy GammaShapeRate (q_out::Any, q_α::PointMass, q_β::Any) = begin
    loggammamean(q_α) - mean(q_α) * logmean(q_β) - (mean(q_α) - 1.0) * logmean(q_out) + mean(q_β) * mean(q_out)
end

@average_energy GammaShapeRate (q_out::Any, q_α::GammaDistributionsFamily, q_β::Any) = begin
    meanlogmean(q_α) - mean(q_α) + 0.5 * (log2π - logmean(q_α)) - mean(q_α) * logmean(q_β) - (mean(q_α) - 1.0) * logmean(q_out) + mean(q_β) * mean(q_out)
end
