export rule

@rule NormalMixture((:m, k), Marginalisation) (q_out::Any, q_switch::Any, q_p::GammaDistributionsFamily) = begin
    pv    = probvec(q_switch)
    T     = eltype(pv)
    z_bar = clamp.(pv, tiny, one(T) - tiny)
    return NormalMeanVariance(mean(q_out), inv(z_bar[k] * mean(q_p)))
end

@rule NormalMixture((:m, k), Marginalisation) (q_out::Any, q_switch::Any, q_p::Wishart) = begin
    pv    = probvec(q_switch)
    T     = eltype(pv)
    z_bar = clamp.(pv, tiny, one(T) - tiny)
    return MvNormalMeanCovariance(mean(q_out), cholinv(z_bar[k] * mean(q_p)))
end
