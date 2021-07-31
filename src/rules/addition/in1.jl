# PointMass
@rule typeof(+)(:in1, Marginalisation) (m_out::PointMass, m_in2::PointMass) = PointMass(mean(m_out) - mean(m_in2))
@rule typeof(+)(:in1, Marginalisation) (m_out::PointMass, m_in2::NormalDistributionsFamily) = NormalMeanVariance(mean(m_out) - mean(m_in2), var(m_in2))
@rule typeof(+)(:in1, Marginalisation) (m_out::PointMass, m_in2::NormalMeanPrecision) = NormalMeanPrecision(mean(m_out) - mean(m_in2), precision(m_in2))
@rule typeof(+)(:in1, Marginalisation) (m_out::PointMass, m_in2::UnivariateNormalDistributionsFamily) = NormalMeanVariance(mean(m_out) - mean(m_in2), var(m_in2))
@rule typeof(+)(:in1, Marginalisation) (m_out::PointMass, m_in2::MultivariateNormalDistributionsFamily) = MvNormalMeanCovariance(mean(m_out) - mean(m_in2), cov(m_in2))
@rule typeof(+)(:in1, Marginalisation) (m_out::PointMass, m_in2::MvNormalMeanPrecision) = MvNormalMeanCovariance(mean(m_out) - mean(m_in2), cov(m_in2))


# NormalDistributionsFamily
@rule typeof(+)(:in1, Marginalisation) (m_out::NormalDistributionsFamily, m_in2::PointMass) = NormalMeanVariance(mean(m_out) - mean(m_in2), var(m_out))
@rule typeof(+)(:in1, Marginalisation) (m_out::NormalDistributionsFamily, m_in2::NormalDistributionsFamily) = NormalMeanVariance(mean(m_out) - mean(m_in2), var(m_out) + var(m_in2))
@rule typeof(+)(:in1, Marginalisation) (m_out::NormalDistributionsFamily, m_in2::NormalMeanPrecision) = NormalMeanVariance(mean(m_out) - mean(m_in2), var(m_out) + var(m_in2))
@rule typeof(+)(:in1, Marginalisation) (m_out::NormalDistributionsFamily, m_in2::UnivariateNormalDistributionsFamily) = NormalMeanVariance(mean(m_out) - mean(m_in2), var(m_out) + var(m_in2))
@rule typeof(+)(:in1, Marginalisation) (m_out::NormalDistributionsFamily, m_in2::MultivariateNormalDistributionsFamily) = MvNormalMeanCovariance(mean(m_out) + mean(m_in2), var(m_out) + var(m_in2))
@rule typeof(+)(:in1, Marginalisation) (m_out::NormalDistributionsFamily, m_in2::MvNormalMeanPrecision) = MvNormalMeanCovariance(mean(m_out) + mean(m_in2), var(m_out) + var(m_in2))


# UnivariateNormalDistributionsFamily
@rule typeof(+)(:in1, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in2::PointMass) = NormalMeanVariance(mean(m_out) - mean(m_in2), var(m_out))
@rule typeof(+)(:in1, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in2::NormalDistributionsFamily) = NormalMeanVariance(mean(m_out) - mean(m_in2), var(m_out))
@rule typeof(+)(:in1, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in2::NormalMeanPrecision) = NormalMeanVariance(mean(m_out) - mean(m_in2), var(m_in2) + var(m_out))
@rule typeof(+)(:in1, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in2::UnivariateNormalDistributionsFamily) = NormalMeanVariance(mean(m_out) - mean(m_in2), var(m_out) + var(m_in2))
@rule typeof(+)(:in1, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in2::MultivariateNormalDistributionsFamily) = MvNormalMeanCovariance(mean(m_out) + mean(m_in2), var(m_out) + var(m_in2))
@rule typeof(+)(:in1, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in2::MvNormalMeanPrecision) = MvNormalMeanCovariance(mean(m_out) + mean(m_in2), var(m_out) + var(m_in2))


# MultivariateNormalDistributionsFamily
@rule typeof(+)(:in1, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in2::PointMass) = MvNormalMeanCovariance(mean(m_out) - mean(m_in2), cov(m_out))
@rule typeof(+)(:in1, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in2::NormalDistributionsFamily) = MvNormalMeanCovariance(mean(m_out) - mean(m_in2), var(m_out))
@rule typeof(+)(:in1, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in2::NormalMeanPrecision) = MvNormalMeanCovariance(mean(m_out) - mean(m_in2), var(m_out) + var(m_in2))
@rule typeof(+)(:in1, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in2::UnivariateNormalDistributionsFamily) = MvNormalMeanCovariance(mean(m_out) - mean(m_in2), var(m_out) + var(m_in2))
@rule typeof(+)(:in1, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in2::MultivariateNormalDistributionsFamily) = MvNormalMeanCovariance(mean(m_out) + mean(m_in2), var(m_out) + var(m_in2))
@rule typeof(+)(:in1, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in2::MvNormalMeanPrecision) = MvNormalMeanCovariance(mean(m_out) + mean(m_in2), var(m_out) + var(m_in2))


# First argument precision --> return precision parameterization

# NormalMeanPrecision
@rule typeof(+)(:in1, Marginalisation) (m_out::NormalMeanPrecision, m_in2::PointMass) = NormalMeanPrecision(mean(m_out) - mean(m_in2), precision(m_out))
@rule typeof(+)(:in1, Marginalisation) (m_out::NormalMeanPrecision, m_in2::NormalDistributionsFamily) = NormalMeanPrecision(mean(m_out) - mean(m_in2), precision(var(m_out) + var(m_in2)))
@rule typeof(+)(:in1, Marginalisation) (m_out::NormalMeanPrecision, m_in2::NormalMeanPrecision) = NormalMeanPrecision(mean(m_out) - mean(m_in2), precision(var(m_out) + var(m_in2)))
@rule typeof(+)(:in1, Marginalisation) (m_out::NormalMeanPrecision, m_in2::UnivariateNormalDistributionsFamily) = NormalMeanPrecision(mean(m_out) - mean(m_in2), precision(var(m_out) + var(m_in2)))
@rule typeof(+)(:in1, Marginalisation) (m_out::NormalMeanPrecision, m_in2::MultivariateNormalDistributionsFamily) = MvNormalMeanPrecision(mean(m_out) - mean(m_in2), precision(var(m_out) + var(m_in2)))
@rule typeof(+)(:in1, Marginalisation) (m_out::NormalMeanPrecision, m_in2::MvNormalMeanPrecision) = MvNormalMeanPrecision(mean(m_out) - mean(m_in2), precision(var(m_out) + var(m_in2)))


# MvNormalMeanPrecision
@rule typeof(+)(:in1, Marginalisation) (m_out::MvNormalMeanPrecision, m_in2::PointMass) = MvNormalMeanPrecision(mean(m_out) - mean(m_in2), precision(m_out))
@rule typeof(+)(:in1, Marginalisation) (m_out::MvNormalMeanPrecision, m_in2::NormalDistributionsFamily) = MvNormalMeanPrecision(mean(m_out) - mean(m_in2), precision(var(m_out) + var(m_in2)))
@rule typeof(+)(:in1, Marginalisation) (m_out::MvNormalMeanPrecision, m_in2::NormalMeanPrecision) = MvNormalMeanPrecision(mean(m_out) - mean(m_in2), precision(var(m_out) + var(m_in2)))
@rule typeof(+)(:in1, Marginalisation) (m_out::MvNormalMeanPrecision, m_in2::UnivariateNormalDistributionsFamily) = MvNormalMeanPrecision(mean(m_out) - mean(m_in2), precision(var(m_out) + var(m_in2)))
@rule typeof(+)(:in1, Marginalisation) (m_out::MvNormalMeanPrecision, m_in2::MultivariateNormalDistributionsFamily) = MvNormalMeanPrecision(mean(m_out) - mean(m_in2), precision(var(m_out) + var(m_in2)))
@rule typeof(+)(:in1, Marginalisation) (m_out::MvNormalMeanPrecision, m_in2::MvNormalMeanPrecision) = MvNormalMeanPrecision(mean(m_out) - mean(m_in2), precision(var(m_out) + var(m_in2)))


@rule typeof(+)(:in1, Marginalisation) (m_out::NormalMeanPrecision, m_in2::NormalMeanPrecision) = begin
    mout, pout = mean_precision(m_out)
    min2, pin2 = mean_precision(m_in2)
    return NormalMeanPrecision(mout - min2, pout * pin2 / (pout + pin2))
end

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