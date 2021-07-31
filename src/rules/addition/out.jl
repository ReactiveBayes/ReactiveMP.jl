# PointMass
@rule typeof(+)(:out, Marginalisation) (m_in1::PointMass, m_in2::PointMass) = PointMass(mean(m_in1) + mean(m_in2))
@rule typeof(+)(:out, Marginalisation) (m_in1::PointMass, m_in2::NormalDistributionsFamily) = NormalMeanVariance(mean(m_in1) + mean(m_in2), var(m_in2))
@rule typeof(+)(:out, Marginalisation) (m_in1::PointMass, m_in2::NormalMeanPrecision) = NormalMeanPrecision(mean(m_in1) + mean(m_in2), precision(m_in2))
@rule typeof(+)(:out, Marginalisation) (m_in1::PointMass, m_in2::UnivariateNormalDistributionsFamily) = NormalMeanVariance(mean(m_in1) + mean(m_in2), var(m_in2))
@rule typeof(+)(:out, Marginalisation) (m_in1::PointMass, m_in2::MultivariateNormalDistributionsFamily) = MvNormalMeanCovariance(mean(m_in1) + mean(m_in2), cov(m_in2))
@rule typeof(+)(:out, Marginalisation) (m_in1::PointMass, m_in2::MvNormalMeanPrecision) = MvNormalMeanCovariance(mean(m_in1) + mean(m_in2), cov(m_in2))


# NormalDistributionsFamily
@rule typeof(+)(:out, Marginalisation) (m_in1::NormalDistributionsFamily, m_in2::PointMass) = NormalMeanVariance(mean(m_in1) + mean(m_in2), var(m_in1))
@rule typeof(+)(:out, Marginalisation) (m_in1::NormalDistributionsFamily, m_in2::NormalDistributionsFamily) = NormalMeanVariance(mean(m_in1) + mean(m_in2), var(m_in1) + var(m_in2))
@rule typeof(+)(:out, Marginalisation) (m_in1::NormalDistributionsFamily, m_in2::NormalMeanPrecision) = NormalMeanVariance(mean(m_in1) + mean(m_in2), var(m_in1) + var(m_in2))
@rule typeof(+)(:out, Marginalisation) (m_in1::NormalDistributionsFamily, m_in2::UnivariateNormalDistributionsFamily) = NormalMeanVariance(mean(m_in1) + mean(m_in2), var(m_in1) + var(m_in2))
@rule typeof(+)(:out, Marginalisation) (m_in1::NormalDistributionsFamily, m_in2::MultivariateNormalDistributionsFamily) = MvNormalMeanCovariance(mean(m_in1) + mean(m_in2), var(m_in1) + var(m_in2))
@rule typeof(+)(:out, Marginalisation) (m_in1::NormalDistributionsFamily, m_in2::MvNormalMeanPrecision) = MvNormalMeanCovariance(mean(m_in1) + mean(m_in2), var(m_in1) + var(m_in2))


# UnivariateNormalDistributionsFamily
@rule typeof(+)(:out, Marginalisation) (m_in1::UnivariateNormalDistributionsFamily, m_in2::PointMass) = NormalMeanVariance(mean(m_in1) + mean(m_in2), var(m_in1))
@rule typeof(+)(:out, Marginalisation) (m_in1::UnivariateNormalDistributionsFamily, m_in2::NormalDistributionsFamily) = NormalMeanVariance(mean(m_in1) + mean(m_in2), var(m_in1) + var(m_in1))
@rule typeof(+)(:out, Marginalisation) (m_in1::UnivariateNormalDistributionsFamily, m_in2::NormalMeanPrecision) = NormalMeanVariance(mean(m_in1) + mean(m_in2), var(m_in1) + var(m_in1))
@rule typeof(+)(:out, Marginalisation) (m_in1::UnivariateNormalDistributionsFamily, m_in2::UnivariateNormalDistributionsFamily) = NormalMeanVariance(mean(m_in1) + mean(m_in2), var(m_in1) + var(m_in1))
@rule typeof(+)(:out, Marginalisation) (m_in1::UnivariateNormalDistributionsFamily, m_in2::MultivariateNormalDistributionsFamily) = MvNormalMeanCovariance(mean(m_in1) + mean(m_in2), var(m_in1) + var(m_in1))
@rule typeof(+)(:out, Marginalisation) (m_in1::UnivariateNormalDistributionsFamily, m_in2::MvNormalMeanPrecision) = MvNormalMeanCovariance(mean(m_in1) + mean(m_in2), var(m_in1) + var(m_in1))


# UnivariateNormalDistributionsFamily
@rule typeof(+)(:out, Marginalisation) (m_in1::MultivariateNormalDistributionsFamily, m_in2::PointMass) = MvNormalMeanCovariance(mean(m_in1) + mean(m_in2), var(m_in1))
@rule typeof(+)(:out, Marginalisation) (m_in1::MultivariateNormalDistributionsFamily, m_in2::NormalDistributionsFamily) = MvNormalMeanCovariance(mean(m_in1) + mean(m_in2), var(m_in1) + var(m_in1))
@rule typeof(+)(:out, Marginalisation) (m_in1::MultivariateNormalDistributionsFamily, m_in2::NormalMeanPrecision) = MvNormalMeanCovariance(mean(m_in1) + mean(m_in2), var(m_in1) + var(m_in1))
@rule typeof(+)(:out, Marginalisation) (m_in1::MultivariateNormalDistributionsFamily, m_in2::UnivariateNormalDistributionsFamily) = MvNormalMeanCovariance(mean(m_in1) + mean(m_in2), var(m_in1) + var(m_in1))
@rule typeof(+)(:out, Marginalisation) (m_in1::MultivariateNormalDistributionsFamily, m_in2::MultivariateNormalDistributionsFamily) = MvNormalMeanCovariance(mean(m_in1) + mean(m_in2), var(m_in1) + var(m_in1))
@rule typeof(+)(:out, Marginalisation) (m_in1::MultivariateNormalDistributionsFamily, m_in2::MvNormalMeanPrecision) = MvNormalMeanCovariance(mean(m_in1) + mean(m_in2), var(m_in1) + var(m_in1))


# First argument w/ precision --> return precision parameterization

# NormalMeanPrecision
@rule typeof(+)(:out, Marginalisation) (m_in1::NormalMeanPrecision, m_in2::PointMass) = NormalMeanPrecision(mean(m_in1) + mean(m_in2), precision(m_in1))
@rule typeof(+)(:out, Marginalisation) (m_in1::NormalMeanPrecision, m_in2::NormalDistributionsFamily) = NormalMeanPrecision(mean(m_in1) + mean(m_in2), precision(var(m_in1) + var(m_in2)))
@rule typeof(+)(:out, Marginalisation) (m_in1::NormalMeanPrecision, m_in2::NormalMeanPrecision) = NormalMeanPrecision(mean(m_in1) + mean(m_in2), precision(var(m_in1) + var(m_in2)))
@rule typeof(+)(:out, Marginalisation) (m_in1::NormalMeanPrecision, m_in2::UnivariateNormalDistributionsFamily) = NormalMeanPrecision(mean(m_in1) + mean(m_in2), precision(var(m_in1) + var(m_in2)))
@rule typeof(+)(:out, Marginalisation) (m_in1::NormalMeanPrecision, m_in2::MultivariateNormalDistributionsFamily) = MvNormalMeanPrecision(mean(m_in1) + mean(m_in2), precision(var(m_in1) + var(m_in2)))
@rule typeof(+)(:out, Marginalisation) (m_in1::NormalMeanPrecision, m_in2::MvNormalMeanPrecision) = MvNormalMeanPrecision(mean(m_in1) + mean(m_in2), precision(var(m_in1) + var(m_in2)))


# MvNormalMeanPrecision
@rule typeof(+)(:out, Marginalisation) (m_in1::MvNormalMeanPrecision, m_in2::PointMass) = MvNormalMeanPrecision(mean(m_in1) + mean(m_in2), precision(m_in1))
@rule typeof(+)(:out, Marginalisation) (m_in1::MvNormalMeanPrecision, m_in2::NormalDistributionsFamily) = MvNormalMeanPrecision(mean(m_in1) + mean(m_in2), precision(var(m_in1) + var(m_in2)))
@rule typeof(+)(:out, Marginalisation) (m_in1::MvNormalMeanPrecision, m_in2::NormalMeanPrecision) = MvNormalMeanPrecision(mean(m_in1) + mean(m_in2), precision(var(m_in1) + var(m_in2)))
@rule typeof(+)(:out, Marginalisation) (m_in1::MvNormalMeanPrecision, m_in2::UnivariateNormalDistributionsFamily) = MvNormalMeanPrecision(mean(m_in1) + mean(m_in2), precision(var(m_in1) + var(m_in2)))
@rule typeof(+)(:out, Marginalisation) (m_in1::MvNormalMeanPrecision, m_in2::MultivariateNormalDistributionsFamily) = MvNormalMeanPrecision(mean(m_in1) + mean(m_in2), precision(var(m_in1) + var(m_in2)))
@rule typeof(+)(:out, Marginalisation) (m_in1::MvNormalMeanPrecision, m_in2::MvNormalMeanPrecision) = MvNormalMeanPrecision(mean(m_in1) + mean(m_in2), precision(var(m_in1) + var(m_in2)))


@rule typeof(+)(:out, Marginalisation) (m_in1::NormalMeanPrecision, m_in2::NormalMeanPrecision) = begin
    min1, pin1 = mean_precision(m_in1)
    min2, pin2 = mean_precision(m_in2)
    return NormalMeanPrecision(min1 + min2, pin1 * pin2 / (pin1 + pin2))
end

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