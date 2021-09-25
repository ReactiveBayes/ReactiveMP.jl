export rule

@rule typeof(dot)(:in1, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in2::PointMass{ <: AbstractVector }, meta::AbstractCorrection) = begin
    x  = mean(m_in2)
    xi = x * weightedmean(m_out)
    w  = correction!(meta, x * precision(m_out) * x')
    return MvNormalWeightedMeanPrecision(xi, w)
end

@rule typeof(dot)(:in1, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in2::PointMass, meta::AbstractCorrection) = begin
    x  = mean(m_in2)
    xi = x * weightedmean(m_out)
    w  = correction!(meta, x * precision(m_out) * x')
    return NormalWeightedMeanPrecision(xi, w)
end

@rule typeof(dot)(:in1, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in2::PointMass) = begin
    x  = mean(m_in2)
    xi = x * weightedmean(m_out)
    w  = x * precision(m_out) * x'
    return NormalWeightedMeanPrecision(xi, w)
end

@rule typeof(dot)(:in1, Marginalisation) (m_out::PointMass{ <: AbstractVector }, m_in2::UnivariateNormalDistributionsFamily, meta::AbstractCorrection) = begin
    x  = mean(m_out)
    xi = x * weightedmean(m_in2)
    w  = correction!(meta, x * precision(m_in2) * x')
    return MvNormalWeightedMeanPrecision(xi, w)
end

@rule typeof(dot)(:in1, Marginalisation) (m_out::PointMass, m_in2::UnivariateNormalDistributionsFamily, meta::AbstractCorrection) = begin
    x  = mean(m_out)
    xi = x * weightedmean(m_in2)
    w  = correction!(meta, x * precision(m_in2) * x')
    return NormalWeightedMeanPrecision(xi, w)
end

@rule typeof(dot)(:in1, Marginalisation) (m_out::PointMass, m_in2::UnivariateNormalDistributionsFamily) = begin
    x  = mean(m_out)
    xi = x * weightedmean(m_in2)
    w  = x * precision(m_in2) * x'
    return NormalWeightedMeanPrecision(xi, w)
end
