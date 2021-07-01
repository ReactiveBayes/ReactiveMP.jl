export rule

@rule typeof(dot)(:in2, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in1::PointMass{ <: AbstractVector }, meta::AbstractCorrection) = begin
    x  = mean(m_in1)
    xi = x * weightedmean(m_out)
    w  = correction!(meta, x * precision(m_out) * x')
    return MvGaussianWeightedMeanPrecision(xi, w)
end
