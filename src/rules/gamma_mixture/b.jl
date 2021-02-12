
@rule GammaMixture((:b, k), Marginalisation) (q_out::Any, q_switch::Any, q_a::NTuple{N1, GammaDistributionsFamily }, q_b::NTuple{N2, GammaDistributionsFamily }) where { N1, N2 } = begin
    return Gamma(1, 1)
end