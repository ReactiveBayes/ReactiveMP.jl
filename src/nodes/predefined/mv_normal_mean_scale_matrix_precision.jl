export MvNormalMeanScaleMatrixPrecision

import StatsFuns: log2π

struct MvNormalMeanScaleMatrixPrecision{T <: Real, M <: AbstractVector{T}, H <: AbstractMatrix{T}} <: AbstractMvNormal
    μ::M
    γ::T
    G::H
end

@node MvNormalMeanScaleMatrixPrecision Stochastic [out, (μ, aliases = [mean]), (γ, aliases = [scale]), (G, aliases = [matrix])]

# default method for mean-field assumption
@average_energy MvNormalMeanScaleMatrixPrecision (q_out::Any, q_μ::Any, q_γ::Any, q_G::Any) = begin
    dim = ndims(q_out)

    m_mean, v_mean = mean_cov(q_μ)
    m_out, v_out   = mean_cov(q_out)
    m_Λ           = mean(q_γ) * mean(q_G)

    result = zero(promote_samplefloattype(q_out, q_μ, q_γ, q_G))
    result += dim * log2π
    result -= dim * mean(log, q_γ)
    result -= dim * mean(logdet, q_G)
    @inbounds for k1 in 1:dim, k2 in 1:dim
        # optimize trace operation (indices can be interchanges because of symmetry)
        result += m_Λ[k1, k2] * (v_out[k1, k2] + v_mean[k1, k2] + (m_out[k2] - m_mean[k2]) * (m_out[k1] - m_mean[k1]))
    end
    result /= 2

    return result
end

# default method for structured mean-field assumption
@average_energy MvNormalMeanScaleMatrixPrecision (q_out_μ::Any, q_γ::Any, q_G::Any) = begin
    dim = div(ndims(q_out_μ), 2)

    m, V = mean_cov(q_out_μ)
    m_out = m[1:dim]
    m_mean = m[dim + 1:end]
    # slice V into blocks according to dim-lengths of out and μ
    v_out = V[1:dim, 1:dim]
    v_mean = V[dim + 1:end, dim + 1:end]
    v_out_mean = V[1:dim, dim + 1:end]
    v_mean_out = V[dim + 1:end, 1:dim]
    m_Λ = mean(q_γ) * mean(q_G)

    result = zero(promote_samplefloattype(q_out_μ, q_γ, q_G))
    result += dim * log2π
    result -= dim * mean(log, q_γ)
    result -= dim * mean(logdet, q_G)
    @inbounds for k1 in 1:dim, k2 in 1:dim
        # optimize trace operation (indices can be interchanges because of symmetry)
        result += m_Λ[k1, k2] * (v_out[k1, k2] - v_out_mean[k1, k2] - v_mean_out[k1, k2] + v_mean[k1, k2] + (m_out[k2] - m_mean[k2]) * (m_out[k1] - m_mean[k1]))
    end
    result /= 2

    return result
end
