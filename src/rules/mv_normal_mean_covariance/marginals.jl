@marginalrule(
    form => Type{ <: MvNormalMeanCovariance },
    on   => :out_mean_covariance,
    messages => (m_out::MvNormalMeanCovariance, m_mean::Dirac, m_covariance::Dirac),
    marginals => Nothing,
    meta => Nothing,
    begin
        return (prod(ProdPreserveParametrisation(MvNormalMeanCovariance(mean(m_mean), mean(m_covariance))), m_out), m_mean, m_covariance)
    end
)

@marginalrule(
    form => Type{ <: MvNormalMeanCovariance },
    on   => :out_mean_covariance,
    messages => (m_out::Dirac, m_mean::MvNormalMeanCovariance, m_covariance::Dirac),
    marginals => Nothing,
    meta => Nothing,
    begin
        return (m_out, prod(ProdPreserveParametrisation(), m_mean, MvNormalMeanCovariance(mean(m_out), mean(m_covariance))), m_covariance)
    end
)

@marginalrule(
    form => Type{ <: MvNormalMeanCovariance },
    on   => :out_mean,
    messages => (m_out::MvNormalMeanCovariance, m_mean::MvNormalMeanCovariance),
    marginals => (q_covariance::Dirac, ),
    meta => Nothing,
    begin
        W_y  = invcov(m_out)
        xi_y = W_y * mean(m_out)

        W_m  = invcov(m_mean)
        xi_m = W_m * mean(m_mean)

        W_bar = cholinv(mean(q_covariance))
        
        xi = [ xi_y; xi_m ]
        W  = [ W_y+W_bar -W_bar; -W_bar W_m+W_bar ]
        
        Σ = cholinv(W)
        μ = Σ * xi
        
        return MvNormalMeanCovariance(μ, Σ)
    end
)