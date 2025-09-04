# Variational                       # 
# --------------------------------- #
@rule MvNormalMeanScaleMatrixPrecision(:γ, Marginalisation) (q_out::Any, q_μ::Any, q_G::Any) = begin
    m_out, v_out   = mean_cov(q_out)
    m_mean, v_mean = mean_cov(q_μ)
    G_bar = mean(q_G)

    α = ndims(q_μ) / 2 + 1
    β = tr( G_bar * (v_mean + v_out + (m_out - m_mean) * (m_out - m_mean)')) / 2

    return GammaShapeRate(convert(eltype(β), α), β)
end

@rule MvNormalMeanScaleMatrixPrecision(:γ, Marginalisation) (q_out_μ::Any, q_G::Any) = begin
    m_out_μ, v_out_μ = mean_cov(q_out_μ)
    G_bar = mean(q_G)

    d = div(ndims(q_out_μ), 2)

    α = d / 2 + 1

    mdiff = @views m_out_μ[1:d] - m_out_μ[(d + 1):end]
    vdiff = @views v_out_μ[1:d, 1:d] - v_out_μ[1:d, (d + 1):end] - v_out_μ[(d + 1):end, 1:d] + v_out_μ[(d + 1):end, (d + 1):end]
    β = tr(G_bar * (vdiff + mdiff * mdiff')) / 2

    return GammaShapeRate(convert(eltype(β), α), β)
end
