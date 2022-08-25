
@node MvNormalMeanCovariance Stochastic [out, (μ, aliases = [mean]), (Σ, aliases = [cov])]

conjugate_type(::Type{<:MvNormalMeanCovariance}, ::Type{Val{:out}}) = MvNormalMeanCovariance
conjugate_type(::Type{<:MvNormalMeanCovariance}, ::Type{Val{:μ}})   = MvNormalMeanCovariance
conjugate_type(::Type{<:MvNormalMeanCovariance}, ::Type{Val{:Σ}})   = InverseWishart

# default method for mean-field assumption
@average_energy MvNormalMeanCovariance (q_out::Any, q_μ::Any, q_Σ::Any) = begin
    # naive: (ndims(q_out) * log2π + mean(logdet, q_Σ) + tr(cholinv(mean(q_Σ))*(v_out + v_mean + (m_out - m_mean)*(m_out - m_mean)'))) / 2
    dim = ndims(q_out)

    m_mean, v_mean = mean_cov(q_μ)
    m_out, v_out   = mean_cov(q_out)
    inv_m_Σ        = mean(cholinv, q_Σ)

    result = zero(promote_type(eltype(q_out), eltype(q_μ), eltype(q_Σ)))
    result += mean(logdet, q_Σ)
    result += dim * log2π
    @inbounds for k1 in 1:dim, k2 in 1:dim   # optimize trace operation (indices can be interchanges because of symmetry)
        result +=
            inv_m_Σ[k1, k2] * (v_out[k1, k2] + v_mean[k1, k2] + (m_out[k2] - m_mean[k2]) * (m_out[k1] - m_mean[k1]))
    end
    result /= 2

    return result
end

# default method for structured mean-field assumption
@average_energy MvNormalMeanCovariance (q_out_μ::Any, q_Σ::Any) = begin
    # naive: @views (d*log2π + mean(logdet, q_Σ) + tr(cholinv(mean(q_Σ))*( V[1:d,1:d] - V[1:d,d+1:end] - V[d+1:end,1:d] + V[d+1:end,d+1:end] + (m[1:d] - m[d+1:end])*(m[1:d] - m[d+1:end])' ))) / 2
    dim = div(ndims(q_out_μ), 2)

    m, V = mean_cov(q_out_μ)
    inv_m_Σ = mean(cholinv, q_Σ)

    result = zero(promote_type(eltype(q_out_μ), eltype(q_Σ)))
    result += mean(logdet, q_Σ)
    result += dim * log2π
    @inbounds for k1 in 1:dim, k2 in 1:dim   # optimize trace operation (indices can be interchanges because of symmetry)
        result +=
            inv_m_Σ[k1, k2] * (
                V[k1, k2] + V[dim+k1, dim+k2] - V[dim+k1, k2] - V[k1, dim+k2] +
                (m[k1] - m[dim+k1]) * (m[k2] - m[dim+k2])
            )
    end
    result /= 2

    return result
end
