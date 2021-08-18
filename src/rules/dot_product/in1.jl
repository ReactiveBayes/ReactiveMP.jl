export rule

@rule typeof(dot)(:in1, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in2::PointMass{ <: AbstractVector }) = begin
    x = mean(m_in2)
    xi = x * weightedmean(m_out)
    w = x * precision(m_out) * x'
    return MvGaussianWeightedMeanPrecision(xi, w)
end
