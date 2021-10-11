
@rule typeof(dot)(:in2, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in1::PointMass, meta::AbstractCorrection) = begin
    A  = mean(m_in1)
    out_wmean, out_prec = weightedmean_precision(m_out)
    
    ξ = A * out_wmean
    W = correction!(meta, A * out_prec * A')
    
    return convert(promote_variate_type(variate_form(m_in1), NormalWeightedMeanPrecision), ξ, W)
end

