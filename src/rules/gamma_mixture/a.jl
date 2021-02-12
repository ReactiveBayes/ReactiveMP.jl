@rule GammaMixture((:a, k), Marginalisation) (q_out::Any, q_switch::Any, q_a::NTuple{N1, GammaDistributionsFamily }, q_b::NTuple{N2, GammaDistributionsFamily}) where { N1, N2 } = begin
    # z_bar = clamp.(probvec(q_switch), tiny, 1.0 - tiny)
    # return MvNormalMeanCovariance(mean(q_out), cholinv(z_bar[k] * mean(q_p[k])))
    # error(3)
    return Gamma(1, 1)
end