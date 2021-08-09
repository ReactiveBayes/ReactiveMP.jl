
# specific case with pointmass inputs
@rule typeof(+)(:in2, Marginalisation) (m_out::PointMass, m_in1::PointMass) = PointMass(mean(m_out) - mean(m_in1))

# specific cases for mean-covariance parameterisation
@rule typeof(+)(:in2, Marginalisation) (m_out::NormalMeanVariance, m_in1::PointMass) = NormalMeanVariance(mean(m_out) - mean(m_in1), var(m_out))
@rule typeof(+)(:in2, Marginalisation) (m_out::PointMass, m_in1::NormalMeanVariance) = NormalMeanVariance(mean(m_out) - mean(m_in1), var(m_in1))

@rule typeof(+)(:in2, Marginalisation) (m_out::MvNormalMeanCovariance, m_in1::PointMass) = begin
    mout, vout = mean_cov(m_out)
    return MvNormalMeanCovariance(m_out - mean(m_in1), vout)
end
@rule typeof(+)(:in2, Marginalisation) (m_out::PointMass, m_in1::MvNormalMeanCovariance) = begin
    min1, vin1 = mean_cov(min1)
    return MvNormalMeanCovariance(mean(m_out) - min1, vin1)
end

# Specific cases for mean-precision parametrisation
@rule typeof(+)(:in2, Marginalisation) (m_out::NormalMeanPrecision, m_in1::PointMass) = NormalMeanPrecision(mean(m_out) - mean(m_in1), precision(m_out))
@rule typeof(+)(:in2, Marginalisation) (m_out::PointMass, m_in1::NormalMeanPrecision) = NormalMeanPrecision(mean(m_out) - mean(m_in1), precision(m_in1))

@rule typeof(+)(:in2, Marginalisation) (m_out::MvNormalMeanPrecision, m_in1::PointMass) = MvNormalMeanPrecision(mean(m_out) - mean(m_in1), precision(m_out))
@rule typeof(+)(:in2, Marginalisation) (m_out::PointMass, m_in1::MvNormalMeanPrecision) = MvNormalMeanPrecision(mean(m_out) - mean(m_in1), precision(m_in1))

# specific cases for weightedmean-precision parameterisation
# note: here we always return the mean-precision parameterization, although the optimal distribution depends on the following nodes
@rule typeof(+)(:in2, Marginalisation) (m_out::NormalWeightedMeanPrecision, m_in1::PointMass) = NormalMeanPrecision(mean(m_out) - mean(m_in1), precision(m_out))
@rule typeof(+)(:in2, Marginalisation) (m_out::PointMass, m_in1::NormalWeightedMeanPrecision) = NormalMeanPrecision(mean(m_out) - mean(m_in1), precision(m_in1))

@rule typeof(+)(:in2, Marginalisation) (m_out::MvNormalWeightedMeanPrecision, m_in1::PointMass) = begin
    mout, wout = mean_precision(m_out)
    return MvNormalMeanPrecision(m_out - mean(m_in1), wout)
end
@rule typeof(+)(:in2, Marginalisation) (m_out::PointMass, m_in1::MvNormalWeightedMeanPrecision) = begin
    min1, win1 = mean_precision(min1)
    return MvNormalMeanPrecision(mean(m_out) - min1, win1)
end

# most generic rules when both inputs are Normal distributions
@rule typeof(+)(:in2, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in1::UnivariateNormalDistributionsFamily) = begin
    mout, vout = mean_var(m_out)
    min1, vin1 = mean_var(m_in1)
    return NormalMeanVariance(mout - min1, vout + vin1)
end

@rule typeof(+)(:in2, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in1::MultivariateNormalDistributionsFamily) = begin
    mout, vout = mean_cov(m_out)
    min1, vin1 = mean_cov(m_in1)
    return MvNormalMeanCovariance(mout - min1, vout + vin1)
end