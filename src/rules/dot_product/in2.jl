export rule

@rule typeof(dot)(:in2, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in1::PointMass{ <: AbstractVector }, meta::AbstractCorrection) = begin
    x  = mean(m_in1)
    xi = x * weightedmean(m_out)
    w  = correction!(meta, x * precision(m_out) * x')
    return MvNormalWeightedMeanPrecision(xi, w)
end

@rule typeof(dot)(:in2, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in1::PointMass, meta::AbstractCorrection) = begin
    x  = mean(m_in1)
    xi = x * weightedmean(m_out)
    w  = correction!(meta, x * precision(m_out) * x')
    return NormalWeightedMeanPrecision(xi, w)
end

@rule typeof(dot)(:in2, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in1::PointMass) = begin
    x  = mean(m_in1)
    xi = x * weightedmean(m_out)
    w  = x * precision(m_out) * x'
    return NormalWeightedMeanPrecision(xi, w)
end

@rule typeof(dot)(:in2, Marginalisation) (m_out::PointMass{ <: AbstractVector }, m_in1::UnivariateNormalDistributionsFamily, meta::AbstractCorrection) = begin
    x  = mean(m_out)
    xi = x * weightedmean(m_in1)
    w  = correction!(meta, x * precision(m_in1) * x')
    return MvNormalWeightedMeanPrecision(xi, w)
end

@rule typeof(dot)(:in2, Marginalisation) (m_out::PointMass, m_in1::UnivariateNormalDistributionsFamily, meta::AbstractCorrection) = begin
    x  = mean(m_out)
    xi = x * weightedmean(m_in1)
    w  = correction!(meta, x * precision(m_in1) * x')
    return NormalWeightedMeanPrecision(xi, w)
end

@rule typeof(dot)(:in2, Marginalisation) (m_out::PointMass, m_in1::UnivariateNormalDistributionsFamily) = begin
    x  = mean(m_out)
    xi = x * weightedmean(m_in1)
    w  = x * precision(m_in1) * x'
    return NormalWeightedMeanPrecision(xi, w)
end
