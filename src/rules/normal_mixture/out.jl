
@rule NormalMixture{N}(:out, Marginalisation) (q_switch::Any, q_m::ManyOf{N, Any}, q_p::ManyOf{N, Any}) where {N} = begin
    πs = probvec(q_switch)

    # Better to preinitialize
    q_p_m = mean.(q_p)
    q_m_m = mean.(q_m)

    W = mapreduce(x -> x[1] * x[2], +, zip(πs, q_p_m))
    ξ = mapreduce(x -> x[1] * x[2] * x[3], +, zip(πs, q_p_m, q_m_m))

    F = variate_form(ξ)

    return convert(promote_variate_type(F, NormalWeightedMeanPrecision), ξ, W)
end
