export rule

@rule NormalMixture((:m, k), Marginalisation) (q_out::Any, q_switch::Any, q_p::Any) = begin
    pv    = probvec(q_switch)
    T     = eltype(pv)
    z_bar = clamp.(pv, tiny, one(T) - tiny)

    F = variate_form(q_out)

    return convert(promote_variate_type(F, NormalMeanPrecision), mean(q_out), z_bar[k] * mean(q_p))
end
