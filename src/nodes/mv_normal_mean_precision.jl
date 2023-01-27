import StatsFuns: log2π

@node MvNormalMeanPrecision Stochastic [out, (μ, aliases = [mean]), (Λ, aliases = [invcov, precision])]

# default method for mean-field assumption
@average_energy MvNormalMeanPrecision (q_out::Any, q_μ::Any, q_Λ::Any) = begin
    # naive: (ndims(q_out) * log2π - mean(logdet, q_Λ) + tr(mean(q_Λ)*(v_out + v_mean + (m_out - m_mean)*(m_out - m_mean)'))) / 2
    dim = ndims(q_out)

    m_mean, v_mean = mean_cov(q_μ)
    m_out, v_out   = mean_cov(q_out)
    m_Λ            = mean(q_Λ)

    result = zero(promote_samplefloattype(q_out, q_μ, q_Λ))
    result += dim * log2π
    result -= mean(logdet, q_Λ)
    @inbounds for k1 in 1:dim, k2 in 1:dim
        # optimize trace operation (indices can be interchanges because of symmetry)
        result += m_Λ[k1, k2] * (v_out[k1, k2] + v_mean[k1, k2] + (m_out[k2] - m_mean[k2]) * (m_out[k1] - m_mean[k1]))
    end
    result /= 2

    return result
end

# specialized method for mean-field assumption with q_Λ::Wishart
@average_energy MvNormalMeanPrecision (q_out::Any, q_μ::Any, q_Λ::Wishart) = begin
    # m_out, v_out = mean_cov(q_out)
    # m_mean, v_mean = mean_cov(q_μ)
    # return (ndims(q_out) * log2π - mean(logdet, q_Λ) + tr(mean(q_Λ)*(v_out + v_mean + (m_out - m_mean)*(m_out - m_mean)'))) / 2
    dim = ndims(q_out)

    m_mean, v_mean = mean_cov(q_μ)
    m_out, v_out   = mean_cov(q_out)
    df_Λ, S_Λ      = params(q_Λ)  # prevent allocation of mean matrix

    T = promote_samplefloattype(q_out, q_μ, q_Λ)
    result = zero(T)

    @inbounds for k1 in 1:dim, k2 in 1:dim
        # optimize trace operation (indices can be interchanges because of symmetry)
        result += S_Λ[k1, k2] * (v_out[k1, k2] + v_mean[k1, k2] + (m_out[k2] - m_mean[k2]) * (m_out[k1] - m_mean[k1]))
    end

    result *= df_Λ
    result += dim * convert(T, log2π)
    result -= mean(logdet, q_Λ)
    result /= 2

    return result
end

# default method for structured mean-field assumption
@average_energy MvNormalMeanPrecision (q_out_μ::Any, q_Λ::Any) = begin
    # naive: @views (d*log2π - mean(logdet, q_Λ) + tr(mean(q_Λ)*( V[1:d,1:d] - V[1:d,d+1:end] - V[d+1:end,1:d] + V[d+1:end,d+1:end] + (m[1:d] - m[d+1:end])*(m[1:d] - m[d+1:end])' ))) / 2
    dim = div(ndims(q_out_μ), 2)

    m, V = mean_cov(q_out_μ)
    m_Λ  = mean(q_Λ)

    T = promote_samplefloattype(q_out_μ, q_Λ)

    result = zero(T)
    result += dim * convert(T, log2π)
    result -= mean(logdet, q_Λ)
    @inbounds for k1 in 1:dim, k2 in 1:dim
        # optimize trace operation (indices can be interchanges because of symmetry)
        result += m_Λ[k1, k2] * (V[k1, k2] + V[dim + k1, dim + k2] - V[dim + k1, k2] - V[k1, dim + k2] + (m[k1] - m[dim + k1]) * (m[k2] - m[dim + k2]))
    end
    result /= 2

    return result
end

# specialized method for structured mean-field assumption with q_Λ::Wishart
@average_energy MvNormalMeanPrecision (q_out_μ::Any, q_Λ::Wishart) = begin
    # naive: (ndims(q_out) * log2π - mean(logdet, q_Λ) + tr(mean(q_Λ)*(v_out + v_mean + (m_out - m_mean)*(m_out - m_mean)'))) / 2
    dim = div(ndims(q_out_μ), 2)

    m, V      = mean_cov(q_out_μ)
    df_Λ, S_Λ = params(q_Λ)     # prevent allocation of mean matrix

    result = zero(promote_samplefloattype(q_out_μ, q_Λ))

    @inbounds for k1 in 1:dim, k2 in 1:dim
        # optimize trace operation (indices can be interchanges because of symmetry)
        result += S_Λ[k1, k2] * (V[k1, k2] + V[dim + k1, dim + k2] - V[dim + k1, k2] - V[k1, dim + k2] + (m[k1] - m[dim + k1]) * (m[k2] - m[dim + k2]))
    end
    result *= df_Λ
    result += dim * log2π
    result -= mean(logdet, q_Λ)
    result /= 2

    return result
end
