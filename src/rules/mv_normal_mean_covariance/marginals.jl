export marginalrule

@marginalrule MvNormalMeanCovariance(:out_μ_Σ) (m_out::MultivariateNormalDistributionsFamily, m_μ::PointMass, m_Σ::PointMass) = begin
    return (out = prod(ProdPreserveParametrisation(), MvNormalMeanCovariance(mean(m_μ), mean(m_Σ)), m_out), μ = m_μ, Σ = m_Σ)
end   

@marginalrule MvNormalMeanCovariance(:out_μ_Σ) (m_out::PointMass, m_μ::MvNormalMeanCovariance, m_Σ::PointMass) = begin
    return (out = m_out, μ = prod(ProdPreserveParametrisation(), m_μ, MvNormalMeanCovariance(mean(m_μ), mean(m_Σ))), Σ = m_Σ)
end

@marginalrule MvNormalMeanCovariance(:out_μ) (m_out::MvNormalMeanCovariance, m_μ::MvNormalMeanCovariance, q_Σ::Any) = begin
    W_y  = invcov(m_out)
    xi_y = W_y * mean(m_out)

    W_m  = invcov(m_μ)
    xi_m = W_m * mean(m_μ)

    W_bar = cholinv(mean(q_Σ))
    
    xi = [ xi_y; xi_m ]
    W  = [ W_y+W_bar -W_bar; -W_bar W_m+W_bar ]
    
    Σ = cholinv(W)
    μ = Σ * xi
    
    return MvNormalMeanCovariance(μ, Σ)
end

@marginalrule MvNormalMeanCovariance(:out_μ_Σ) (m_out::MultivariateNormalDistributionsFamily, m_μ::MultivariateNormalDistributionsFamily, m_Σ::PointMass) = begin
    W_y  = invcov(m_out)
    xi_y = W_y * mean(m_out)

    W_m  = invcov(m_μ)
    xi_m = W_m * mean(m_μ)

    W_bar = cholinv(mean(m_Σ))
    
    xi = [ xi_y; xi_m ]
    W  = [ W_y+W_bar -W_bar; -W_bar W_m+W_bar ]
    
    Σ = cholinv(W)
    μ = Σ * xi

    return (out_μ = MvNormalMeanCovariance(μ, Σ), Σ = m_Σ)
end

@marginalrule MvNormalMeanCovariance(:out_μ) (m_out::PointMass, m_μ::MvNormalMeanCovariance, q_Σ::Any) = begin
    return (out = m_out, μ = prod(ProdPreserveParametrisation(), MvNormalMeanCovariance(mean(m_out), mean(q_Σ)), m_μ))
end