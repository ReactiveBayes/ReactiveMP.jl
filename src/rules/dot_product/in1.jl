
@rule typeof(dot)(:in1, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in2::PointMass, meta::Union{AbstractCorrectionStrategy, Nothing}) = begin
    return @call_rule typeof(dot)(:in2, Marginalisation) (m_out = m_out, m_in1 = m_in2, meta = meta)
end

@rule typeof(dot)(:in1, Marginalisation) (m_out::UnivariateNormalDistributionsFamily, m_in2::NormalDistributionsFamily, meta::Union{AbstractCorrectionStrategy, Nothing}) = begin
    return error("The rule for the dot product node between two NormalDistributionsFamily instances is not available in closed form. Please use SoftDot instead.")
end
