# PointMass
@rule typeof(+)(:in2, Marginalisation) (m_out::PointMass, m_in1::PointMass) = PointMass(mean(m_out) - mean(m_in1))
@rule typeof(+)(:in2, Marginalisation) (m_out::PointMass, m_in1::NormalDistributionsFamily) = NormalMeanVariance(mean(m_out) - mean(m_in1), var(m_in1))
@rule typeof(+)(:in2, Marginalisation) (m_out::PointMass, m_in1::NormalMeanPrecision) = NormalMeanPrecision(mean(m_out) - mean(m_in1), precision(m_in1))
@rule typeof(+)(:in2, Marginalisation) (m_out::PointMass, m_in1::UnivariateNormalDistributionsFamily) = NormalMeanVariance(mean(m_out) - mean(m_in1), var(m_in1))
@rule typeof(+)(:in2, Marginalisation) (m_out::PointMass, m_in1::MultivariateNormalDistributionsFamily) = MvNormalMeanCovariance(mean(m_out) - mean(m_in1), cov(m_in1))
@rule typeof(+)(:in2, Marginalisation) (m_out::PointMass, m_in1::MvNormalMeanPrecision) = MvNormalMeanCovariance(mean(m_out) - mean(m_in1), cov(m_in1))


# NormalDistributionsFamily
@rule typeof(+)(:in2, Marginalisation) (m_out::NormalDistributionsFamily, m_in1::NormalDistributionsFamily) = NormalMeanVariance(mean(m_out) - mean(m_in1), var(m_out) + var(m_in1))
@rule typeof(+)(:in2, Marginalisation) (m_out::NormalDistributionsFamily, m_in1::PointMass) = NormalMeanVariance(mean(m_out) - mean(m_in1), var(m_out))
@rule typeof(+)(:in2, Marginalisation) (m_out::NormalDistributionsFamily, m_in1::NormalMeanPrecision) = NormalMeanVariance(mean(m_out) - mean(m_in1), var(m_out) + var(m_in1))
@rule typeof(+)(:in2, Marginalisation) (m_out::NormalDistributionsFamily, m_in1::UnivariateNormalDistributionsFamily) = NormalMeanVariance(mean(m_out) - mean(m_in1), var(m_out) + var(m_in1))
@rule typeof(+)(:in2, Marginalisation) (m_out::NormalDistributionsFamily, m_in1::MultivariateNormalDistributionsFamily) = MvNormalMeanCovariance(mean(m_out) - mean(m_in1), var(m_out) + var(m_in1))
@rule typeof(+)(:in2, Marginalisation) (m_out::NormalDistributionsFamily, m_in1::MvNormalMeanPrecision) = MvNormalMeanCovariance(mean(m_out) - mean(m_in1), var(m_out) + var(m_in1))


# UnivariateNormalDistributionsFamily
@rule typeof(+)(:in2, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in1::PointMass) = NormalMeanVariance(mean(m_out) - mean(m_in1), var(m_out))
@rule typeof(+)(:in2, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in1::NormalDistributionsFamily) = NormalMeanVariance(mean(m_out) - mean(m_in1), var(m_out))
@rule typeof(+)(:in2, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in1::NormalMeanPrecision) = NormalMeanVariance(mean(m_out) - mean(m_in1), var(m_in1) + var(m_out))
@rule typeof(+)(:in2, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in1::UnivariateNormalDistributionsFamily) = NormalMeanVariance(mean(m_out) - mean(m_in1), var(m_out) + var(m_in1))
@rule typeof(+)(:in2, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in1::MultivariateNormalDistributionsFamily) = MvNormalMeanCovariance(mean(m_out) - mean(m_in1), var(m_out) + var(m_in1))
@rule typeof(+)(:in2, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in1::MvNormalMeanPrecision) = MvNormalMeanCovariance(mean(m_out) - mean(m_in1), var(m_out) + var(m_in1))


# MultivariateNormalDistributionsFamily
@rule typeof(+)(:in2, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in1::PointMass) = MvNormalMeanCovariance(mean(m_out) - mean(m_in1), cov(m_out))
@rule typeof(+)(:in2, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in1::NormalDistributionsFamily) = MvNormalMeanCovariance(mean(m_out) - mean(m_in1), var(m_out))
@rule typeof(+)(:in2, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in1::NormalMeanPrecision) = MvNormalMeanCovariance(mean(m_out) - mean(m_in1), var(m_out) + var(m_in1))
@rule typeof(+)(:in2, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in1::UnivariateNormalDistributionsFamily) = MvNormalMeanCovariance(mean(m_out) - mean(m_in2), var(m_out) + var(m_in1))
@rule typeof(+)(:in2, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in1::MultivariateNormalDistributionsFamily) = MvNormalMeanCovariance(mean(m_out) - mean(m_in2), var(m_out) + var(m_in1))
@rule typeof(+)(:in2, Marginalisation) (m_out::MultivariateNormalDistributionsFamily, m_in1::MvNormalMeanPrecision) = MvNormalMeanCovariance(mean(m_out) - mean(m_in2), var(m_out) + var(m_in1))


# First argument precision --> return precision parameterization

# NormalMeanPrecision
@rule typeof(+)(:in2, Marginalisation) (m_out::NormalMeanPrecision, m_in1::PointMass) = NormalMeanPrecision(mean(m_out) - mean(m_in1), precision(m_out))
@rule typeof(+)(:in2, Marginalisation) (m_out::NormalMeanPrecision, m_in1::NormalDistributionsFamily) = NormalMeanPrecision(mean(m_out) - mean(m_in1), precision(var(m_out) + var(m_in1)))
@rule typeof(+)(:in2, Marginalisation) (m_out::NormalMeanPrecision, m_in1::NormalMeanPrecision) = NormalMeanPrecision(mean(m_out) - mean(m_in1), precision(var(m_out) + var(m_in2)))
@rule typeof(+)(:in2, Marginalisation) (m_out::NormalMeanPrecision, m_in1::UnivariateNormalDistributionsFamily) = NormalMeanPrecision(mean(m_out) - mean(m_in1), precision(var(m_out) + var(m_in1)))
@rule typeof(+)(:in2, Marginalisation) (m_out::NormalMeanPrecision, m_in1::MultivariateNormalDistributionsFamily) = MvNormalMeanPrecision(mean(m_out) - mean(m_in1), precision(var(m_out) + var(m_in1)))
@rule typeof(+)(:in2, Marginalisation) (m_out::NormalMeanPrecision, m_in1::MvNormalMeanPrecision) = MvNormalMeanPrecision(mean(m_out) - mean(m_in1), precision(var(m_out) + var(m_in1)))


# NormalMeanPrecision
@rule typeof(+)(:in2, Marginalisation) (m_out::MvNormalMeanPrecision, m_in1::PointMass) = MvNormalMeanPrecision(mean(m_out) - mean(m_in1), precision(m_out))
@rule typeof(+)(:in2, Marginalisation) (m_out::MvNormalMeanPrecision, m_in1::NormalDistributionsFamily) = MvNormalMeanPrecision(mean(m_out) - mean(m_in1), precision(var(m_out) + var(m_in1)))
@rule typeof(+)(:in2, Marginalisation) (m_out::MvNormalMeanPrecision, m_in1::NormalMeanPrecision) = MvNormalMeanPrecision(mean(m_out) - mean(m_in1), precision(var(m_out) + var(m_in2)))
@rule typeof(+)(:in2, Marginalisation) (m_out::MvNormalMeanPrecision, m_in1::UnivariateNormalDistributionsFamily) = MvNormalMeanPrecision(mean(m_out) - mean(m_in1), precision(var(m_out) + var(m_in1)))
@rule typeof(+)(:in2, Marginalisation) (m_out::MvNormalMeanPrecision, m_in1::MultivariateNormalDistributionsFamily) = MvNormalMeanPrecision(mean(m_out) - mean(m_in1), precision(var(m_out) + var(m_in1)))
@rule typeof(+)(:in2, Marginalisation) (m_out::MvNormalMeanPrecision, m_in1::MvNormalMeanPrecision) = MvNormalMeanPrecision(mean(m_out) - mean(m_in1), precision(var(m_out) + var(m_in1)))


@rule typeof(+)(:in2, Marginalisation) (m_out::NormalMeanPrecision, m_in1::NormalMeanPrecision) = begin
    mout, pout = mean_precision(m_out)
    min1, pin1 = mean_precision(m_in1)
    return NormalMeanPrecision(mout - min1, pout * pin1 / (pout + pin1))
end

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