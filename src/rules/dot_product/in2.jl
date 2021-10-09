export rule

@rule typeof(dot)(:in2, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in1::PointMass{ <: AbstractVector }, meta::AbstractCorrection) = begin
    x  = mean(m_in1)
    xi, w = weightedmean_precision(m_out)
    xi = x * xi
    w  = correction!(meta, x * w * x')
    return MvNormalWeightedMeanPrecision(xi, w)
end

@rule typeof(dot)(:in2, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in1::PointMass, meta::AbstractCorrection) = begin
    x  = mean(m_in1)
    xi, w = weightedmean_precision(m_out)
    xi = x * xi
    w  = correction!(meta, x * w * x')
    return NormalWeightedMeanPrecision(xi, w)
end

@rule typeof(dot)(:in2, Marginalisation) (m_out::PointMass{ <: AbstractVector }, m_in1::UnivariateNormalDistributionsFamily, meta::AbstractCorrection) = begin
    x  = mean(m_out)
    xi, w = weightedmean_precision(m_in1)
    xi = x * xi
    w  = correction!(meta, x * w * x')
    return MvNormalWeightedMeanPrecision(xi, w)
end

@rule typeof(dot)(:in2, Marginalisation) (m_out::PointMass, m_in1::UnivariateNormalDistributionsFamily, meta::AbstractCorrection) = begin
    x  = mean(m_out)
    xi, w = weightedmean_precision(m_in1)
    xi = x * xi
    w  = correction!(meta, x * w * x')
    return NormalWeightedMeanPrecision(xi, w)
end

