
@rule typeof(dot)(:out, Marginalisation) (m_in1::NormalDistributionsFamily, m_in2::PointMass, meta::Union{AbstractCorrectionStrategy, Nothing}) = begin
    return @call_rule typeof(dot)(:out, Marginalisation) (m_in1 = m_in2, m_in2 = m_in1, meta = meta)
end

@rule typeof(dot)(:out, Marginalisation) (m_in1::PointMass, m_in2::NormalDistributionsFamily, meta::Union{AbstractCorrectionStrategy, Nothing}) = begin
    A = mean(m_in1)
    in2_mean, in2_cov = mean_cov(m_in2)
    return NormalMeanVariance(dot(A, in2_mean), dot(A, in2_cov, A))
end

@rule typeof(dot)(:out, Marginalisation) (m_in1::NormalDistributionsFamily, m_in2::NormalDistributionsFamily, meta::Union{AbstractCorrectionStrategy, Nothing}) = begin
    return error("The rule for the dot product node between two NormalDistributionsFamily instances is not available in closed form. Please use SoftDot instead.")
end
