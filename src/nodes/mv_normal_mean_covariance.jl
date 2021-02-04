
@node MvNormalMeanCovariance Stochastic [ out, (μ, aliases = [ mean ]), (Σ, aliases = [ cov ]) ]

conjugate_type(::Type{ <: MvNormalMeanCovariance }, ::Type{ Val{ :out } }) = MvNormalMeanCovariance
conjugate_type(::Type{ <: MvNormalMeanCovariance }, ::Type{ Val{ :μ } })   = MvNormalMeanCovariance
conjugate_type(::Type{ <: MvNormalMeanCovariance }, ::Type{ Val{ :Σ } })   = InverseWishart

@average_energy MvNormalMeanCovariance (q_out::Any, q_μ::Any, q_Σ::Any) = begin
    (m_mean, v_mean) = mean(q_μ), cov(q_μ)
        (m_out, v_out)   = mean(q_out), cov(q_out)

        0.5 * (ndims(q_out) * log2π + logdet(mean(q_Σ)) + tr(cholinv(mean(q_Σ))*(v_out + v_mean + (m_out - m_mean)*(m_out - m_mean)')))
end

@average_energy MvNormalMeanCovariance (q_out_μ::Any, q_Σ::Any) = begin
    (m, V) = mean(q_out_μ), cov(q_out_μ)
    d = Int64(ndims(q_out_μ)/2)
    @views 0.5*(d*log2π + logdet(mean(q_Σ)) + tr(cholinv(mean(q_Σ))*( V[1:d,1:d] - V[1:d,d+1:end] - V[d+1:end,1:d] + V[d+1:end,d+1:end] + (m[1:d] - m[d+1:end])*(m[1:d] - m[d+1:end])' ) ))
end