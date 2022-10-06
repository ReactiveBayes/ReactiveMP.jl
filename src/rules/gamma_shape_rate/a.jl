import DomainSets

@rule GammaShapeRate(:α, Marginalisation) (q_out::Gamma, q_β::Gamma) = begin
    return ContinuousUnivariateLogPdf(
        DomainSets.HalfLine(),
        (α) -> α * mean(log, q_β) + (α - 1) * mean(log, q_out) - loggamma(α)
    )
end
