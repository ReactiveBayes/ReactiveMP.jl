
@node MvNormalMeanCovariance Stochastic [ out, (μ, aliases = [ mean ]), (Σ, aliases = [ cov ]) ]

conjugate_type(::Type{ <: MvNormalMeanCovariance }, ::Type{ Val{ :out } }) = MvNormalMeanCovariance
conjugate_type(::Type{ <: MvNormalMeanCovariance }, ::Type{ Val{ :μ } })   = MvNormalMeanCovariance
conjugate_type(::Type{ <: MvNormalMeanCovariance }, ::Type{ Val{ :Σ } })   = InverseWishart

@average_energy MvNormalMeanCovariance (q_out::Any, q_μ::Any, q_Σ::Any) = begin
    m_mean, v_mean = mean_cov(q_μ)
    m_out, v_out   = mean_cov(q_out)
    return (ndims(q_out) * log2π + logdet(mean(q_Σ)) + tr(cholinv(mean(q_Σ))*(v_out + v_mean + (m_out - m_mean)*(m_out - m_mean)'))) / 2
end

@average_energy MvNormalMeanCovariance (q_out_μ::Any, q_Σ::Any) = begin
    m, V = mean_cov(q_out_μ)
    d = div(ndims(q_out_μ), 2)
    return @views (d*log2π + logdet(mean(q_Σ)) + tr(cholinv(mean(q_Σ))*( V[1:d,1:d] - V[1:d,d+1:end] - V[d+1:end,1:d] + V[d+1:end,d+1:end] + (m[1:d] - m[d+1:end])*(m[1:d] - m[d+1:end])' ))) / 2
end