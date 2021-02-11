
@node GammaShapeRate Stochastic [ out, (α, aliases = [ shape ]), (β, aliases = [ rate ]) ]

@average_energy GammaShapeRate (q_out::Any, q_α::Any, q_β::Any) = begin
    return labsgamma(mean(q_α)) - mean(q_α) * logmean(q_β) - (mean(q_α) - 1.0) * logmean(q_out) + mean(q_β) * mean(q_out)
end
