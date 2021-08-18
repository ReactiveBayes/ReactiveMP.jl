export rule

@rule NormalMixture((:m, k), Marginalisation) (q_out::Any, q_switch::Any, q_p::GammaDistributionsFamily) = begin
    z_bar = clamp.(probvec(q_switch), tiny, 1.0 - tiny)
    return NormalMeanVariance(mean(q_out), inv(z_bar[k] * mean(q_p)))
end

@rule NormalMixture((:m, k), Marginalisation) (q_out::Any, q_switch::Any, q_p::Wishart) = begin
    z_bar = clamp.(probvec(q_switch), tiny, 1.0 - tiny)
    return MvNormalMeanCovariance(mean(q_out), cholinv(z_bar[k] * mean(q_p)))
end