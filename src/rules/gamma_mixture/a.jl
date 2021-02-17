
@rule GammaMixture((:a, k), Marginalisation) (q_out::Any, q_switch::Any, q_a::NTuple{N1, GammaDistributionsFamily }, q_b::NTuple{N2, GammaDistributionsFamily}, meta::GammaMixtureNodeMetadata) where { N1, N2 } = begin
    p = probvec(q_switch)[k]
    β = logmean(q_out) + logmean(q_b[k])
    γ = p * β
    return GammaShapeLikelihood(p, γ, get_shape_likelihood_approximation(meta))
end