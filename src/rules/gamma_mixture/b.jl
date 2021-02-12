@rule GammaMixture((:b, k), Marginalisation) (q_out::Any, q_switch::Any, q_a::NTuple{N1, GammaDistributionsFamily }, q_b::NTuple{N2, GammaDistributionsFamily }) where { N1, N2 } = begin
    π_k = probvec(q_switch)[k]
    return GammaShapeRate(1 + π_k*mean(q_a[k]), π_k*mean(q_out))
end
