export rule

@rule typeof(dot)(:in2, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in1::PointMass{ <: AbstractVector }) = begin
    x = mean(m_in1)
    xi = x * weightedmean(m_out)
    w = x * precision(m_out) * x'
    return MvGaussianWeightedMeanPrecision(xi, w)
end
