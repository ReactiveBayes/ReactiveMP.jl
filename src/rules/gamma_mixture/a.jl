import SpecialFunctions: digamma

@rule GammaMixture((:a, k), Marginalisation) (q_out::Any, q_switch::Any, q_a::NTuple{N1, GammaDistributionsFamily }, q_b::NTuple{N2, GammaDistributionsFamily}) where { N1, N2 } = begin
    â_k = mean(q_out)*mean(q_b[k])
    Ψ = logmean(q_out) + logmean(q_b[k]) - digamma(â_k)
    return GammaShapeRate(1, probvec(q_switch)[k]*Ψ)
end
