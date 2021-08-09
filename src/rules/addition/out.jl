
# specific case with pointmass inputs
@rule typeof(+)(:out, Marginalisation) (m_in1::PointMass, m_in2::PointMass) = PointMass(mean(m_in1) + mean(m_in2))

# specific cases for mean-covariance parameterisation
@rule typeof(+)(:out, Marginalisation) (m_in1::PointMass, m_in2::UnivariateNormalDistributionsFamily) = NormalMeanVariance(mean(m_in1) + mean(m_in2), var(m_in2))
@rule typeof(+)(:out, Marginalisation) (m_in1::UnivariateNormalDistributionsFamily, m_in2::PointMass) = NormalMeanVariance(mean(m_in1) + mean(m_in2), var(m_in1))

@rule typeof(+)(:out, Marginalisation) (m_in1::PointMass, m_in2::MultivariateNormalDistributionsFamily) = MvNormalMeanCovariance(mean(m_in1) + mean(m_in2), cov(m_in2))
@rule typeof(+)(:out, Marginalisation) (m_in1::MultivariateNormalDistributionsFamily, m_in2::PointMass) = MvNormalMeanCovariance(mean(m_in1) + mean(m_in2), cov(m_in1))

# Specific cases for mean-precision parametrisation
@rule typeof(+)(:out, Marginalisation) (m_in1::NormalMeanPrecision, m_in2::PointMass) = NormalMeanPrecision(mean(m_in1) + mean(m_in2), precision(m_in1))
@rule typeof(+)(:out, Marginalisation) (m_in1::PointMass, m_in2::NormalMeanPrecision) = NormalMeanPrecision(mean(m_in1) + mean(m_in2), precision(m_in2))

@rule typeof(+)(:out, Marginalisation) (m_in1::MvNormalMeanPrecision, m_in2::PointMass) = MvNormalMeanPrecision(mean(m_in1) + mean(m_in2), precision(m_in1))
@rule typeof(+)(:out, Marginalisation) (m_in1::PointMass, m_in2::MvNormalMeanPrecision) = MvNormalMeanPrecision(mean(m_in1) + mean(m_in2), precision(m_in2))

# specific cases for weightedmean-precision parameterisation
# note: here we always return the mean-precision parameterization, although the optimal distribution depends on the following nodes
@rule typeof(+)(:out, Marginalisation) (m_in1::NormalWeightedMeanPrecision, m_in2::PointMass) = NormalMeanPrecision(mean(m_in1) + mean(m_in2), precision(m_in1))
@rule typeof(+)(:out, Marginalisation) (m_in1::PointMass, m_in2::NormalWeightedMeanPrecision) = NormalMeanPrecision(mean(m_in1) + mean(m_in2), precision(m_in2))

@rule typeof(+)(:out, Marginalisation) (m_in1::MvNormalWeightedMeanPrecision, m_in2::PointMass) = begin
    min1, win1 = mean_precision(m_in1)
    return MvNormalMeanPrecision(min1 + mean(m_in2), win1)
end
@rule typeof(+)(:out, Marginalisation) (m_in1::PointMass, m_in2::MvNormalWeightedMeanPrecision) = begin
    min2, win2 = mean_precision(m_in2)
    return MvNormalMeanPrecision(mean(m_in1) + min2, win2)
end

# most generic rules when both inputs are Normal distributions
@rule typeof(+)(:out, Marginalisation) (m_in1::UnivariateNormalDistributionsFamily, m_in2::UnivariateNormalDistributionsFamily) = begin 
    m1, v1 = mean_var(m_in1)
    m2, v2 = mean_var(m_in2)
    return NormalMeanVariance(m1 + m2, v1 + v2)
end

@rule typeof(+)(:out, Marginalisation) (m_in1::MultivariateNormalDistributionsFamily, m_in2::MultivariateNormalDistributionsFamily) = begin
    m1, v1 = mean_cov(m_in1)
    m2, v2 = mean_cov(m_in2)
    return MvNormalMeanCovariance(m1 + m2, v1 + v2)
end