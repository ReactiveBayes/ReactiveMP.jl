# distritbutions
@rule typeof(+)(:out, Marginalisation) (m_in1::Distribution, m_in2::Distribution) = begin
    return convolve(m_in1, m_in2)
end

@rule typeof(+)(:out, Marginalisation) (m_in1::UnivariateNormalDistributionsFamily, m_in2::UnivariateNormalDistributionsFamily) = begin
    min1, vin1 = mean_var(m_in1)
    min2, vin2 = mean_var(m_in2)
    return NormalMeanVariance(min1 + min2, vin1 + vin2)
end

@rule typeof(+)(:out, Marginalisation) (m_in1::MultivariateNormalDistributionsFamily, m_in2::MultivariateNormalDistributionsFamily) = begin
    min1, vin1 = mean_cov(m_in1)
    min2, vin2 = mean_cov(m_in2)
    return MvNormalMeanCovariance(min1 + min2, vin1 + vin2)
end

# PointMass
@rule typeof(+)(:out, Marginalisation) (m_in1::PointMass, m_in2::PointMass) = PointMass(mean(m_in1) + mean(m_in2))

@rule typeof(+)(:out, Marginalisation) (m_in1::NormalMeanPrecision, m_in2::PointMass) = begin
    min1, win1 = mean_precision(m_in1)
    return NormalMeanPrecision(min1 + mean(m_in2), win1)
end

@rule typeof(+)(:out, Marginalisation) (m_in1::PointMass, m_in2::NormalMeanPrecision) = begin
    return @call_rule typeof(+)(:out, Marginalisation) (m_in1 = m_in2, m_in2 = m_in1, meta = meta)
end

@rule typeof(+)(:out, Marginalisation) (m_in1::MvNormalMeanPrecision, m_in2::PointMass) = begin
    min1, win1 = mean_precision(m_in1)
    return MvNormalMeanPrecision(min1 + mean(m_in2), win1)
end

@rule typeof(+)(:out, Marginalisation) (m_in1::PointMass, m_in2::MvNormalMeanPrecision) = begin
    return @call_rule typeof(+)(:out, Marginalisation) (m_in1 = m_in2, m_in2 = m_in1, meta = meta)
end

@rule typeof(+)(:out, Marginalisation) (m_in1::UnivariateNormalDistributionsFamily, m_in2::PointMass) = begin
    min1, vin1 = mean_var(m_in1)
    return NormalMeanVariance(min1 + mean(m_in2), vin1)
end

@rule typeof(+)(:out, Marginalisation) (m_in1::PointMass, m_in2::UnivariateNormalDistributionsFamily) = begin
    return @call_rule typeof(+)(:out, Marginalisation) (m_in1 = m_in2, m_in2 = m_in1, meta = meta)
end

@rule typeof(+)(:out, Marginalisation) (m_in1::MultivariateNormalDistributionsFamily, m_in2::PointMass) = begin
    min1, vin1 = mean_cov(m_in1)
    return MvNormalMeanCovariance(mean(m_in2) + min1, vin1)
end

@rule typeof(+)(:out, Marginalisation) (m_in1::PointMass, m_in2::MultivariateNormalDistributionsFamily) = begin
    return @call_rule typeof(+)(:out, Marginalisation) (m_in1 = m_in2, m_in2 = m_in1, meta = meta)
end

# specialized
@rule typeof(+)(:out, Marginalisation) (
    m_in1::MvNormalWeightedMeanPrecision{T, Vector{T}, Matrix{T}}, m_in2::MvNormalWeightedMeanPrecision{T, Vector{T}, Matrix{T}}
) where {T <: LinearAlgebra.BlasFloat} = begin
    # `mean_cov` here allocates a new matrix and vector, which can be used later on as scratch space. This is not desirable logic but it is efficient.
    min2, vin2 = mean_cov(m_in2)
    vin1 = cov(m_in1)
    BLAS.gemv!('N', one(T), vin1, weightedmean(m_in1), one(T), min2)
    vin2 .+= vin1
    return MvNormalMeanCovariance(min2, vin2)
end

@rule typeof(+)(:out, Marginalisation) (m_in1::MvNormalWeightedMeanPrecision{T1, Vector{T1}, Matrix{T1}}, m_in2::PointMass) where {T1} = begin
    ξin1, win1 = weightedmean_precision(m_in1)
    ξout = win1 * mean(m_in2)
    ξout .+= ξin1
    return MvNormalWeightedMeanPrecision(ξout, win1)
end

@rule typeof(+)(:out, Marginalisation) (m_in1::PointMass, m_in2::MvNormalWeightedMeanPrecision{T1, Vector{T1}, Matrix{T1}}) where {T1} = begin
    ξin2, win2 = weightedmean_precision(m_in2)
    ξout = win2 * mean(m_in1)
    ξout .+= ξin2
    return MvNormalWeightedMeanPrecision(ξout, win2)
end
