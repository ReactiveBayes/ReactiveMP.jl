# Variational                       # 
# --------------------------------- #
 @rule MvNormalMeanPrecision(:Λ, Marginalisation) (q_out::Any, q_μ::Any) = begin
    m_out, v_out   = mean_cov(q_out)
    m_mean, v_mean = mean_cov(q_μ)
    d = ndims(q_μ)

    df   = d + 2
    invS_raw = v_mean + v_out + (m_mean - m_out) * (m_mean - m_out)'
    invS = invS_raw + 1e-6 * diagm(ones(d))
    
    return WishartFast(df, invS)
end 

@rule MvNormalMeanPrecision(:Λ, Marginalisation) (q_out_μ::Any,) = begin
    m_out_μ, v_out_μ = mean_cov(q_out_μ)

    d = div(ndims(q_out_μ), 2)

    mdiff = @views m_out_μ[1:d] - m_out_μ[(d + 1):end]
    vdiff = @views v_out_μ[1:d, 1:d] - v_out_μ[1:d, (d + 1):end] - v_out_μ[(d + 1):end, 1:d] + v_out_μ[(d + 1):end, (d + 1):end]
    invS  = vdiff + mdiff * mdiff'

    return WishartFast(d + 2, invS)
end
