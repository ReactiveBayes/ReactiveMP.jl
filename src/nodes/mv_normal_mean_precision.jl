import StatsFuns: log2π

@node MvNormalMeanPrecision Stochastic [ out, (μ, aliases = [ mean ]), (Λ, aliases = [ invcov, precision ]) ]

conjugate_type(::Type{ <: MvNormalMeanPrecision }, ::Type{ Val{ :out } }) = MvNormalMeanPrecision
conjugate_type(::Type{ <: MvNormalMeanPrecision }, ::Type{ Val{ :μ } })   = MvNormalMeanPrecision
conjugate_type(::Type{ <: MvNormalMeanPrecision }, ::Type{ Val{ :Λ } })   = Wishart

@average_energy MvNormalMeanPrecision (q_out::Any, q_μ::Any, q_Λ::Any) = begin
    m_mean, v_mean = mean_cov(q_μ)
    m_out, v_out   = mean_cov(q_out)
    return (ndims(q_out) * log2π + logdet(cholinv(mean(q_Λ))) + tr(mean(q_Λ)*(v_out + v_mean + (m_out - m_mean)*(m_out - m_mean)'))) / 2
end   

@average_energy MvNormalMeanPrecision (q_out_μ::Any, q_Λ::Any) = begin
    m, V = mean_cov(q_out_μ)
    d = div(ndims(q_out_μ), 2)
    return @views (d*log2π + logdet(cholinv(mean(q_Λ))) + tr(mean(q_Λ)*( V[1:d,1:d] - V[1:d,d+1:end] - V[d+1:end,1:d] + V[d+1:end,d+1:end] + (m[1:d] - m[d+1:end])*(m[1:d] - m[d+1:end])' ))) / 2
end