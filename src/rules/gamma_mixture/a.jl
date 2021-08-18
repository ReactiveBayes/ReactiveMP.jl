
@rule GammaMixture((:a, k), Marginalisation) (q_out::Any, q_switch::Any, q_b::GammaDistributionsFamily) = begin
    p = probvec(q_switch)[k]
    β = logmean(q_out) + logmean(q_b)
    γ = p * β
    return GammaShapeLikelihood(p, γ)
end
