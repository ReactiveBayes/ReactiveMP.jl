@marginalrule(
    formtype  => MvNormalMeanCovariance,
    on        => :out_μ_Σ,
    messages  => (m_out::MvNormalMeanCovariance, m_μ::Dirac, m_Σ::Dirac),
    marginals => Nothing,
    meta      => Nothing,
    begin
        return (out = prod(ProdPreserveParametrisation(MvNormalMeanCovariance(mean(m_μ), mean(m_Σ))), m_out), μ = m_μ, Σ = m_Σ)
    end
)

@marginalrule(
    formtype  => MvNormalMeanCovariance,
    on        => :out_μ_Σ,
    messages  => (m_out::Dirac, m_μ::MvNormalMeanCovariance, m_Σ::Dirac),
    marginals => Nothing,
    meta      => Nothing,
    begin
        return (out = m_out, μ = prod(ProdPreserveParametrisation(), m_μ, MvNormalMeanCovariance(mean(m_μ), mean(m_Σ))), Σ = m_Σ)
    end
)

@marginalrule(
    formtype  => MvNormalMeanCovariance,
    on        => :out_μ,
    messages  => (m_out::MvNormalMeanCovariance, m_μ::MvNormalMeanCovariance),
    marginals => (q_Σ::Dirac, ),
    meta      => Nothing,
    begin
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
)