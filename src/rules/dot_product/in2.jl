
@rule typeof(dot)(:in2, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in1::PointMass, meta::Union{AbstractCorrectionStrategy, Nothing}) = begin
    A = mean(m_in1)
    out_wmean, out_prec = weightedmean_precision(m_out)

    ξ = A * out_wmean
    W = correction!(meta, v_a_vT(A, out_prec))

    return convert(promote_variate_type(variate_form(typeof(m_in1)), NormalWeightedMeanPrecision), ξ, W)
end

@rule typeof(dot)(:in2, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in1::NormalDistributionsFamily, meta::Union{AbstractCorrectionStrategy, Nothing}) = begin
    return error("The rule for the dot product node between two NormalDistributionsFamily instances is not available in closed form. Please use SoftDot instead.")
end
