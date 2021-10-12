
# Variational                       # 
# --------------------------------- #
@rule MvNormalMeanPrecision(:Λ, Marginalisation) (q_out::Any, q_μ::Any) = begin
    m_out, v_out   = mean_cov(q_out)
    m_mean, v_mean = mean_cov(q_μ)

    df = ndims(q_μ) + 2
    S  = Matrix(Hermitian(cholinv(v_mean + v_out + (m_mean - m_out) * (m_mean - m_out)')))

    return Wishart(df, S)
end

@rule MvNormalMeanPrecision(:Λ, Marginalisation) (q_out_μ::Any, ) = begin
    m_out_μ, v_out_μ   = mean_cov(q_out_μ)

    d = div(ndims(q_out_μ), 2)

    mdiff = @views m_out_μ[1:d] - m_out_μ[d+1:end]
    vdiff = @views v_out_μ[1:d,1:d] - v_out_μ[1:d,d+1:end] - v_out_μ[d+1:end, 1:d] + v_out_μ[d+1:end,d+1:end]
    S     = Matrix(Hermitian(cholinv(vdiff + mdiff * mdiff')))

    return Wishart(d + 2, S)
end