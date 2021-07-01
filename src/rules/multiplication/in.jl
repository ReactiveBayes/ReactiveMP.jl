export rule

@rule typeof(*)(:in, Marginalisation) (m_out::PointMass, m_A::PointMass) = PointMass(mean(m_in1) * mean(m_in2))

@rule typeof(*)(:in, Marginalisation) (m_out::F, m_A::PointMass, meta::AbstractCorrection) where { F <: NormalDistributionsFamily } = begin
    A = mean(m_A)
    W = correction!(meta, A' * precision(m_out) * A)
    return convert(promote_variate_type(F, NormalWeightedMeanPrecision), A' * weightedmean(m_out), W)
end