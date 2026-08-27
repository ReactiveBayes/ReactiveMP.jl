@rule Bilinear(:out, Marginalisation) (
    m_in::UnivariateNormalDistributionsFamily, q_a::PointMass
) = begin
    # m(out) ∝ ∫ exp(a⋅out⋅in) m_in(in) d(in) = exp(a⋅μ_in⋅out + a²⋅v_in⋅out²/2)
    # NOTE: improper (negative precision) — inherent to this factor; downstream
    # posteriors are proper only when incoming precisions dominate (w_out⋅w_in > a²).
    return NormalWeightedMeanPrecision(
        mean(q_a) * mean(m_in), -abs2(mean(q_a)) * var(m_in)
    )
end
