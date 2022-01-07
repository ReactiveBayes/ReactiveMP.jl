import StatsFuns: log2π

@node MvNormalMeanPrecision Stochastic [ out, (μ, aliases = [ mean ]), (Λ, aliases = [ invcov, precision ]) ]

conjugate_type(::Type{ <: MvNormalMeanPrecision }, ::Type{ Val{ :out } }) = MvNormalMeanPrecision
conjugate_type(::Type{ <: MvNormalMeanPrecision }, ::Type{ Val{ :μ } })   = MvNormalMeanPrecision
conjugate_type(::Type{ <: MvNormalMeanPrecision }, ::Type{ Val{ :Λ } })   = Wishart

# default method for mean-field assumption
@average_energy MvNormalMeanPrecision (q_out::Any, q_μ::Any, q_Λ::Any) = begin
    # naive: (ndims(q_out) * log2π + logdet(cholinv(mean(q_Λ))) + tr(mean(q_Λ)*(v_out + v_mean + (m_out - m_mean)*(m_out - m_mean)'))) / 2
    dim = ndims(q_out)

    m_mean, v_mean = mean_cov(q_μ)
    m_out, v_out   = mean_cov(q_out)
    m_Λ            = mean(q_Λ)

    result = dim * log2π
    result -= chollogdet(m_Λ)
    @turbo for k1 ∈ 1:dim, k2 ∈ 1:dim   # optimize trace operation (indices can be interchanges because of symmetry)
        result += m_Λ[k1,k2] * (v_out[k1,k2] + v_mean[k1,k2] + (m_out[k2] - m_mean[k2]) * (m_out[k1] - m_mean[k1]))
    end
    result /= 2

    return result
end   

# specialized method for mean-field assumption with q_Λ::Wishart
@average_energy MvNormalMeanPrecision (q_out::Any, q_μ::Any, q_Λ::Wishart) = begin
    # m_out, v_out = mean_cov(q_out)
    # m_mean, v_mean = mean_cov(q_μ)
    # return (ndims(q_out) * log2π + logdet(cholinv(mean(q_Λ))) + tr(mean(q_Λ)*(v_out + v_mean + (m_out - m_mean)*(m_out - m_mean)'))) / 2
    dim = ndims(q_out)

    m_mean, v_mean = mean_cov(q_μ)
    m_out, v_out   = mean_cov(q_out)
    df_Λ, S_Λ      = q_Λ.df, q_Λ.S.mat  # prevent allocation of mean matrix

    result = 0.0
    @turbo for k1 ∈ 1:dim, k2 ∈ 1:dim   # optimize trace operation (indices can be interchanges because of symmetry)
        result += S_Λ[k1,k2] * (v_out[k1,k2] + v_mean[k1,k2] + (m_out[k2] - m_mean[k2]) * (m_out[k1] - m_mean[k1]))
    end
    result *= df_Λ
    result += dim * log2π
    result -= chollogdet(S_Λ)
    result -= dim*log(df_Λ)
    result /= 2

    return result
end

# default method for structured mean-field assumption
@average_energy MvNormalMeanPrecision (q_out_μ::Any, q_Λ::Any) = begin
    # naive: @views (d*log2π + logdet(cholinv(mean(q_Λ))) + tr(mean(q_Λ)*( V[1:d,1:d] - V[1:d,d+1:end] - V[d+1:end,1:d] + V[d+1:end,d+1:end] + (m[1:d] - m[d+1:end])*(m[1:d] - m[d+1:end])' ))) / 2
    dim = div(ndims(q_out_μ), 2)

    m, V = mean_cov(q_out_μ)
    m_Λ  = mean(q_Λ)

    result = dim * log2π
    result -= chollogdet(m_Λ)
    @turbo for k1 ∈ 1:dim, k2 ∈ 1:dim   # optimize trace operation (indices can be interchanges because of symmetry)
        result += m_Λ[k1,k2] * (V[k1,k2] + V[dim+k1,dim+k2] - V[dim+k1,k2] - V[k1,dim+k2] + (m[k1] - m[dim+k1]) * (m[k2] - m[dim+k2]))
    end
    result /= 2

    return result
end

# specialized method for structured mean-field assumption with q_Λ::Wishart
@average_energy MvNormalMeanPrecision (q_out_μ::Any, q_Λ::Wishart) = begin
    # naive: (ndims(q_out) * log2π + logdet(cholinv(mean(q_Λ))) + tr(mean(q_Λ)*(v_out + v_mean + (m_out - m_mean)*(m_out - m_mean)'))) / 2
    dim = div(ndims(q_out_μ), 2)

    m, V       = mean_cov(q_out_μ)
    df_Λ, S_Λ  = q_Λ.df, q_Λ.S.mat      # prevent allocation of mean matrix

    result = 0.0
    @turbo for k1 ∈ 1:dim, k2 ∈ 1:dim   # optimize trace operation (indices can be interchanges because of symmetry)
        result += S_Λ[k1,k2] * (V[k1,k2] + V[dim+k1,dim+k2] - V[dim+k1,k2] - V[k1,dim+k2] + (m[k1] - m[dim+k1]) * (m[k2] - m[dim+k2]))
    end
    result *= df_Λ
    result += dim * log2π
    result -= chollogdet(S_Λ)
    result -= dim*log(df_Λ)
    result /= 2

    return result
end