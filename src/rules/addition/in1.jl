export rule

# specific case with pointmass inputs
@rule typeof(+)(:in1, Marginalisation) (m_out::PointMass, m_in2::PointMass) = PointMass(mean(m_out) - mean(m_in2))

# specific cases for mean-covariance parameterisation
@rule typeof(+)(:in1, Marginalisation) (m_out::NormalMeanVariance, m_in2::PointMass) = NormalMeanVariance(mean(m_out) - mean(m_in2), var(m_out))
@rule typeof(+)(:in1, Marginalisation) (m_out::PointMass, m_in2::NormalMeanVariance) = NormalMeanVariance(mean(m_out) - mean(m_in2), var(m_in2))

@rule typeof(+)(:in1, Marginalisation) (m_out::MvNormalMeanCovariance, m_in2::PointMass) = begin
    mout, vout = mean_cov(m_out)
    return MvNormalMeanCovariance(m_out - mean(m_in2), vout)
end
@rule typeof(+)(:in1, Marginalisation) (m_out::PointMass, m_in2::MvNormalMeanCovariance) = begin
    min2, vin2 = mean_cov(min2)
    return MvNormalMeanCovariance(mean(m_out) - min2, vin2)
end

# Specific cases for mean-precision parametrisation
@rule typeof(+)(:in1, Marginalisation) (m_out::NormalMeanPrecision, m_in2::PointMass) = NormalMeanPrecision(mean(m_out) - mean(m_in2), precision(m_out))
@rule typeof(+)(:in1, Marginalisation) (m_out::PointMass, m_in2::NormalMeanPrecision) = NormalMeanPrecision(mean(m_out) - mean(m_in2), precision(m_in2))

@rule typeof(+)(:in1, Marginalisation) (m_out::MvNormalMeanPrecision, m_in2::PointMass) = MvNormalMeanPrecision(mean(m_out) - mean(m_in2), precision(m_out))
@rule typeof(+)(:in1, Marginalisation) (m_out::PointMass, m_in2::MvNormalMeanPrecision) = MvNormalMeanPrecision(mean(m_out) - mean(m_in2), precision(m_in2))

# specific cases for weightedmean-precision parameterisation
# note: here we always return the mean-precision parameterization, although the optimal distribution depends on the following nodes
@rule typeof(+)(:in1, Marginalisation) (m_out::NormalWeightedMeanPrecision, m_in2::PointMass) = NormalMeanPrecision(mean(m_out) - mean(m_in2), precision(m_out))
@rule typeof(+)(:in1, Marginalisation) (m_out::PointMass, m_in2::NormalWeightedMeanPrecision) = NormalMeanPrecision(mean(m_out) - mean(m_in2), precision(m_in2))

@rule typeof(+)(:in1, Marginalisation) (m_out::MvNormalWeightedMeanPrecision, m_in2::PointMass) = begin
    mout, wout = mean_precision(m_out)
    return MvNormalMeanPrecision(m_out - mean(m_in2), wout)
end
@rule typeof(+)(:in1, Marginalisation) (m_out::PointMass, m_in2::MvNormalWeightedMeanPrecision) = begin
    min2, win2 = mean_precision(min2)
    return MvNormalMeanPrecision(mean(m_out) - min2, win2)
end

# most generic rules when both inputs are Normal distributions
@rule typeof(+)(:in1, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in2::UnivariateNormalDistributionsFamily) = begin
    mout, vout = mean_var(m_out)
    min2, vin2 = mean_var(m_in2)
    return NormalMeanVariance(mout - min2, vout + vin2)
end

@rule typeof(+)(:in1, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in2::MultivariateNormalDistributionsFamily) = begin
    mout, vout = mean_cov(m_out)
    min2, vin2 = mean_cov(m_in2)
    return MvNormalMeanCovariance(mout - min2, vout + vin2)
end
