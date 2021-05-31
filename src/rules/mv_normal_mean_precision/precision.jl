
# Variational                       # 
# --------------------------------- #
@rule MvNormalMeanPrecision(:Λ, Marginalisation) (q_out::Any, q_μ::Any) = begin
    m_out, v_out   = mean_cov(q_out)
    m_mean, v_mean = mean_cov(q_μ)

    df = ndims(q_μ) + 2
    S  = Matrix(Hermitian(cholinv(v_mean + v_out + (m_mean - m_out) * (m_mean - m_out)')))

    return Wishart(df, S)
end