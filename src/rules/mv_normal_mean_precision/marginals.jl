export marginalrule

@marginalrule MvNormalMeanPrecision(:out_μ_Λ) (m_out::MvNormalMeanPrecision, m_μ::PointMass, m_Λ::PointMass) = begin
    return (out = prod(ProdAnalytical(), MvNormalMeanPrecision(mean(m_μ), mean(m_Λ)), m_out), μ = m_μ, Λ = m_Λ)
end

@marginalrule MvNormalMeanPrecision(:out_μ_Λ) (m_out::PointMass, m_μ::MvNormalMeanPrecision, m_Λ::PointMass) = begin
    return (out = m_out, μ = prod(ProdAnalytical(), m_μ, MvNormalMeanPrecision(mean(m_out), mean(m_Λ))), Λ = m_Λ)
end

@marginalrule MvNormalMeanPrecision(:out_μ_Λ) (m_out::MultivariateNormalDistributionsFamily, m_μ::PointMass, m_Λ::PointMass) = begin 
    return (out = prod(ProdAnalytical(), m_out, MvNormalMeanPrecision(mean(m_μ), mean(m_Λ))), μ = m_μ, Λ = m_Λ)
end

@marginalrule MvNormalMeanPrecision(:out_μ) (m_out::MvNormalMeanPrecision, m_μ::MvNormalMeanPrecision, q_Λ::Any) = begin
    W_y  = invcov(m_out)
    xi_y = W_y * mean(m_out)

    W_m  = invcov(m_μ)
    xi_m = W_m * mean(m_μ)

    W_bar = mean(q_Λ)
    
    Λ  = [ W_y + W_bar -W_bar; -W_bar W_m + W_bar ]
    μ  = cholinv(Λ) * [ xi_y; xi_m ]
    
    return MvNormalMeanPrecision(μ, Λ)
end
