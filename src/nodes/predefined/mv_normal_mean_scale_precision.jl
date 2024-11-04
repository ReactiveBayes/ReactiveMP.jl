export MvNormalMeanScalePrecision

import ExponentialFamily: MvNormalMeanScalePrecision
import StatsFuns: log2π

@node MvNormalMeanScalePrecision Stochastic [out, (μ, aliases = [mean]), (γ, aliases = [precision])]

# default method for mean-field assumption
@average_energy MvNormalMeanScalePrecision (q_out::Any, q_μ::Any, q_γ::Any) = begin
    dim = ndims(q_out)

    m_mean, v_mean = mean_cov(q_μ)
    m_out, v_out   = mean_cov(q_out)
    m_Λ            = mean(q_γ) * diageye(dim)

    result = zero(promote_samplefloattype(q_out, q_μ, q_γ))
    result += dim * log2π
    result -= dim * mean(log, q_γ)
    @inbounds for k1 in 1:dim, k2 in 1:dim
        # optimize trace operation (indices can be interchanges because of symmetry)
        result += m_Λ[k1, k2] * (v_out[k1, k2] + v_mean[k1, k2] + (m_out[k2] - m_mean[k2]) * (m_out[k1] - m_mean[k1]))
    end
    result /= 2

    return result
end

# default method for structured mean-field assumption
@average_energy MvNormalMeanScalePrecision (q_out_μ::Any, q_γ::Any) = begin
    dim = div(ndims(q_out_μ), 2)

    m, V = mean_cov(q_out_μ)
    m_Λ  = mean(q_γ) * diageye(dim)

    result = zero(promote_samplefloattype(q_out_μ, q_γ))
    result += dim * log2π
    result -= dim * mean(log, q_γ)
    @inbounds for k1 in 1:dim, k2 in 1:dim
        # optimize trace operation (indices can be interchanges because of symmetry)
        result += m_Λ[k1, k2] * (V[k1, k2] + V[dim + k1, dim + k2] - V[dim + k1, k2] - V[k1, dim + k2] + (m[k1] - m[dim + k1]) * (m[k2] - m[dim + k2]))
    end
    result /= 2

    return result
end
