
@marginalrule typeof(*)(:A_in) (m_out::NormalDistributionsFamily, m_A::PointMass, m_in::NormalDistributionsFamily, meta::Union{<:AbstractCorrectionStrategy, Nothing}) = begin
    b_in = @call_rule typeof(*)(:in, Marginalisation) (m_out = m_out, m_A = m_A, meta = meta)
    q_in = prod(ClosedProd(), b_in, m_in)
    return (A = m_A, in = q_in)
end

# Specific version for scalar with switched arguments.
# Note that for multivariate case in general multiplication is not a commutative operation, 
# but for scalars we make an exception
@marginalrule typeof(*)(:A_in) (
    m_out::UnivariateNormalDistributionsFamily, m_A::UnivariateNormalDistributionsFamily, m_in::PointMass{<:Real}, meta::Union{<:AbstractCorrectionStrategy, Nothing}
) = begin
    flipped_result = @call_marginalrule typeof(*)(:A_in) (m_out = m_out, m_A = m_in, m_in = m_A, meta = meta)
    return (A = flipped_result[:in], in = flipped_result[:A])
end

# Specific version for univariate and pointmass input
@marginalrule typeof(*)(:A_in) (m_out::MvNormalMeanPrecision, m_A::UnivariateNormalDistributionsFamily, m_in::PointMass{<:AbstractVector}, meta::Any) = begin
    m_outbound_A = @call_rule typeof(*)(:A, Marginalisation) (m_out = m_out, m_in = m_in, meta = meta)
    q_a = prod(ClosedProd(), m_A, m_outbound_A)
    return (A = q_a, in = m_in)
end

