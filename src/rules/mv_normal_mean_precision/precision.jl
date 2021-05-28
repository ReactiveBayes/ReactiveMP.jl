export rule

@rule MvNormalMeanPrecision(:Λ, Marginalisation) (q_out::PointMass, q_μ::MultivariateNormalDistributionsFamily) = begin
    m_mean, v_mean = mean_cov(q_μ)
    m_out, v_out   = mean_cov(q_out)

    df = ndims(q_μ) + 2.0
    S  = Matrix(Hermitian(cholinv(v_mean + v_out + (m_mean - m_out)*(m_mean - m_out)')))

    return Wishart(df, S)
end