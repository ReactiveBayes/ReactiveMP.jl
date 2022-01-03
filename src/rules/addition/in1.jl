# distributions
@rule typeof(+)(:in1, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in2::UnivariateNormalDistributionsFamily) = begin
    min2, vin2 = mean_var(m_in2)
    mout, vout = mean_var(m_out)
    return NormalMeanVariance(mout - min2, vout + vin2)
end

@rule typeof(+)(:in1, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in2::MultivariateNormalDistributionsFamily) = begin
    min2, vin2 = mean_cov(m_in2)
    mout, vout = mean_cov(m_out)
    return MvNormalMeanCovariance(mout - min2, vout + vin2)
end

# PointMass
@rule typeof(+)(:in1, Marginalisation) (m_out::PointMass, m_in2::PointMass) = PointMass(mean(m_out) - mean(m_in2))

@rule typeof(+)(:in1, Marginalisation) (m_out::PointMass, m_in2::NormalMeanPrecision) = begin
    min2, win2 = mean_precision(m_in2)
    return NormalMeanPrecision(mean(m_out) - min2, win2)
end

@rule typeof(+)(:in1, Marginalisation) (m_out::NormalMeanPrecision, m_in2::PointMass) = begin
    mout, wout = mean_precision(m_out)
    return NormalMeanPrecision(mout - mean(m_in2), wout)
end


@rule typeof(+)(:in1, Marginalisation) (m_out::PointMass, m_in2::MvNormalMeanPrecision) = begin
    min2, win2 = mean_precision(m_in2)
    return MvNormalMeanPrecision(mean(m_out) - min2, win2)
end

@rule typeof(+)(:in1, Marginalisation) (m_out::MvNormalMeanPrecision, m_in2::PointMass) = begin
    mout, wout = mean_precision(m_out)
    return MvNormalMeanPrecision(mout - mean(m_in2), wout)
end


@rule typeof(+)(:in1, Marginalisation) (m_out::PointMass, m_in2::UnivariateNormalDistributionsFamily) = begin
    min2, vin2 = mean_var(m_in2)
    return NormalMeanVariance(mean(m_out) - min2, vin2)
end

@rule typeof(+)(:in1, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in2::PointMass) = begin
    mout, vout = mean_var(m_out)
    return NormalMeanVariance(mout - mean(m_in2), vout)
end


@rule typeof(+)(:in1, Marginalisation) (m_out::PointMass, m_in2::MultivariateNormalDistributionsFamily) = begin
    min2, vin2 = mean_cov(m_in2)
    return MvNormalMeanCovariance(mean(m_out) - min2, vin2)
end

@rule typeof(+)(:in1, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in2::PointMass) = begin
    mout, vout = mean_cov(m_out)
    return MvNormalMeanCovariance(mout - mean(m_in2), vout)
end

# specialized
@rule typeof(+)(:in1, Marginalisation) (m_out::MvNormalWeightedMeanPrecision{T1}, m_in2::MvNormalWeightedMeanPrecision{T2}) where { T1 <: LinearAlgebra.BlasFloat, T2 <: LinearAlgebra.BlasFloat } = begin

    min2, vin2 = mean_cov(m_in2)
    vout = cov(m_out)
    T = promote_type(T1, T2)
    BLAS.gemv!('N', -one(T), vout, weightedmean(m_out), one(T), min2)
    vin2 .+= vout
    return MvNormalMeanCovariance(min2, vin2)

end

@rule typeof(+)(:in1, Marginalisation) (m_out::MvNormalWeightedMeanPrecision{T1}, m_in2::PointMass) where { T1 } = begin

    ξout, wout = weightedmean_precision(m_out)
    ξin1 = wout*mean(m_in2)
    ξin1 .-= ξout
    T = promote_type(T1, eltype(m_in2))
    ξin1 .*= -one(T)
    return MvNormalWeightedMeanPrecision(ξin1, wout)

end

@rule typeof(+)(:in1, Marginalisation) (m_out::PointMass, m_in2::MvNormalWeightedMeanPrecision) = begin

    ξin2, win2 = weightedmean_precision(m_in2)
    ξin1 = win2*mean(m_out)
    ξin1 .-= ξin2
    return MvNormalWeightedMeanPrecision(ξin1, win2)

end