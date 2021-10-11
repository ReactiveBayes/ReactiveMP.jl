export marginalrule

@marginalrule MvNormalMeanPrecision(:out_μ_Λ) (m_out::MultivariateNormalDistributionsFamily, m_μ::PointMass, m_Λ::PointMass) = begin
    return (out = prod(ProdAnalytical(), MvNormalMeanPrecision(mean(m_μ), mean(m_Λ)), m_out), μ = m_μ, Λ = m_Λ)
end

@marginalrule MvNormalMeanPrecision(:out_μ_Λ) (m_out::PointMass, m_μ::MultivariateNormalDistributionsFamily, m_Λ::PointMass) = begin
    return (out = m_out, μ = prod(ProdAnalytical(), m_μ, MvNormalMeanPrecision(mean(m_out), mean(m_Λ))), Λ = m_Λ)
end

@marginalrule MvNormalMeanPrecision(:out_μ_Λ) (m_out::MultivariateNormalDistributionsFamily, m_μ::MultivariateNormalDistributionsFamily, m_Λ::PointMass) = begin
    xi_y, W_y  = weightedmean_precision(m_out)
    xi_m, W_m  = weightedmean_precision(m_μ)

    W_bar = mean(q_Λ)
    
    Λ  = [ W_y + W_bar -W_bar; -W_bar W_m + W_bar ]
    ξ  = [ xi_y; xi_m ]
    
    return (out_μ = MvNormalWeightedMeanPrecision(ξ, Λ), Λ = m_Λ)
end

@marginalrule MvNormalMeanPrecision(:out_μ) (m_out::MultivariateNormalDistributionsFamily, m_μ::MultivariateNormalDistributionsFamily, q_Λ::Any) = begin
    xi_y, W_y  = weightedmean_precision(m_out)
    xi_m, W_m  = weightedmean_precision(m_μ)

    W_bar = mean(q_Λ)
    
    Λ  = [ W_y + W_bar -W_bar; -W_bar W_m + W_bar ]
    ξ  = [ xi_y; xi_m ]
    
    return MvNormalWeightedMeanPrecision(ξ, Λ)
end

@marginalrule MvNormalMeanPrecision(:out_μ) (m_out::PointMass, m_μ::MultivariateNormalDistributionsFamily, q_Λ::Any) = begin
    return (out = m_out, μ = prod(ProdAnalytical(), MvNormalMeanPrecision(mean(m_out), mean(q_Λ)), m_μ))
end

@marginalrule MvNormalMeanPrecision(:out_μ) (m_out::MultivariateNormalDistributionsFamily, m_μ::PointMass, q_Λ::Any) = begin
    return (out = prod(ProdAnalytical(), MvNormalMeanPrecision(mean(m_μ), mean(q_Λ)), m_out), μ = m_μ)
end

## ProdFinal / BIFM related

@marginalrule MvNormalMeanPrecision(:out_μ_Λ) (m_out::ProdFinal, m_μ::PointMass, m_Λ::PointMass) = begin 
    return (out = getdist(m_out), μ = m_μ, Λ = m_Λ)
end

@marginalrule MvNormalMeanPrecision(:out_μ_Λ) (m_out::PointMass, m_μ::ProdFinal, m_Λ::PointMass) = begin 
    return (out = m_out, μ = getdist(m_μ), Λ = m_Λ)
end