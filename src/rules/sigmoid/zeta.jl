@rule Sigmoid(:ζ, Marginalisation) (q_out::Any, q_in::UnivariateNormalDistributionsFamily) = begin
    m_in, v_in = mean_var(q_in)
    T = promote_type(eltype(m_in), eltype(v_in))
    return PointMass{T}(sqrt(m_in^2 + v_in))
end
