export marginalrule

@marginalrule NormalMeanPrecision(:out_μ_τ) (m_out::NormalDistributionsFamily, m_μ::PointMass, m_τ::PointMass) = begin
    return (out = prod(ProdAnalytical(), NormalMeanPrecision(mean(m_μ), mean(m_τ)), m_out), μ = m_μ, τ = m_τ)
end

@marginalrule NormalMeanPrecision(:out_μ_τ) (m_out::PointMass, m_μ::NormalDistributionsFamily, m_τ::PointMass) = begin
    return (out = m_out, μ = prod(ProdAnalytical(), m_μ, NormalMeanPrecision(mean(m_out), mean(m_τ))), τ = m_τ)
end

@marginalrule NormalMeanPrecision(:out_μ) (m_out::NormalMeanPrecision, m_μ::NormalMeanPrecision, q_τ::Gamma) = begin
    W_out  = invcov(m_out)
    W_μ    = invcov(m_μ)
    xi_out = W_out * mean(m_out)
    xi_μ   = W_μ * mean(m_μ)

    W_bar = mean(q_τ)

    W = [ W_out + W_bar -W_bar; -W_bar W_μ + W_bar ]
    m = cholinv(W) * [ xi_out; xi_μ ]

    return MvNormalMeanPrecision(m, W)
end