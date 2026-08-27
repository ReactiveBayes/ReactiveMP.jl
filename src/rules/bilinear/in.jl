@rule Bilinear(:in, Marginalisation) (
    m_out::UnivariateNormalDistributionsFamily, q_a::PointMass
) = begin
    # m(in) ∝ ∫ exp(a⋅out⋅in) m_out(out) d(out) = exp(a⋅μ_out⋅in + a²⋅v_out⋅in²/2)
    # NOTE: improper (negative precision) — inherent to this factor; downstream
    # posteriors are proper only when incoming precisions dominate (w_out⋅w_in > a²).
    return NormalWeightedMeanPrecision(mean(q_a) * mean(m_out), -abs2(mean(q_a)) * var(m_out))
end
