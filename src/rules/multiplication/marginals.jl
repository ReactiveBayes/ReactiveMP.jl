
@marginalrule typeof(*)(:A_in) (m_out::NormalDistributionsFamily, m_A::PointMass, m_in::NormalDistributionsFamily, meta::Union{<:AbstractCorrectionStrategy, Nothing}) = begin
    b_in = @call_rule typeof(*)(:in, Marginalisation) (m_out = m_out, m_A = m_A, meta = meta)
    q_in = prod(ProdAnalytical(), b_in, m_in)
    return (A = m_A, in = q_in)
end

# Specific version for scalar with switched arguments.
# Note that for multivariate case in general multiplication is not a commutative operation, 
# but for scalars we make an exception
@marginalrule typeof(*)(:A_in) (
    m_out::UnivariateNormalDistributionsFamily, m_A::UnivariateNormalDistributionsFamily, m_in::PointMass{<:Real}, meta::Union{<:AbstractCorrectionStrategy, Nothing}
) = begin
    return @call_marginalrule typeof(*)(:A_in) (m_out = m_out, m_A = m_in, m_in = m_A, meta = meta)
end
