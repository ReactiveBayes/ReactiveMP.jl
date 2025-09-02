# Variational                       # 
# --------------------------------- #
@rule MvNormalMeanScaleMatrixPrecision(:G, Marginalisation) (q_out::Any, q_μ::Any, q_γ::Any) = begin
    m_out, v_out   = mean_cov(q_out)
    m_mean, v_mean = mean_cov(q_μ)
    γ_bar = mean(q_γ)

    n_G = ndims(q_μ) + 2
    inv_V_G = γ_bar * ( v_out + v_mean + (m_out - m_mean) * (m_out - m_mean)' )

    return WishartFast(convert(eltype(inv_V_G), n_G), inv_V_G)
end

@rule MvNormalMeanScaleMatrixPrecision(:G, Marginalisation) (q_out_μ::Any, q_γ::Any) = begin
    m_out_μ, v_out_μ = mean_cov(q_out_μ)
    γ_bar = mean(q_γ)

    d = div(ndims(q_out_μ), 2)

    n_G = d / 2

    mdiff = @views m_out_μ[1:d] - m_out_μ[(d + 1):end]
    vdiff = @views v_out_μ[1:d, 1:d] - v_out_μ[1:d, (d + 1):end] - v_out_μ[(d + 1):end, 1:d] + v_out_μ[(d + 1):end, (d + 1):end]
    inv_V_G = (vdiff + mdiff * mdiff') * γ_bar

    return WishartFast(convert(eltype(inv_V_G), n_G), inv_V_G)
end
