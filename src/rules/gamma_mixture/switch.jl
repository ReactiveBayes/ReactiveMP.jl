@rule GammaMixture{N}(:switch, Marginalisation) (q_out::Any, q_a::NTuple{N, GammaDistributionsFamily }, q_b::NTuple{N, GammaDistributionsFamily }) where { N } = begin
    # p. 12
    return Categorical([ 0.5, 0.5 ])
end