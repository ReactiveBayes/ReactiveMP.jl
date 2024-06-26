import DomainSets

@rule GammaShapeRate(:α, Marginalisation) (q_out::Any, q_β::GammaDistributionsFamily) = begin
    γ = mean(log, q_β) + mean(log, q_out)
    params = promote(1, γ)
    return GammaShapeLikelihood(params...)
end
