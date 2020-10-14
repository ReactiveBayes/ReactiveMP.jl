@marginalrule(
    form => Type{ <: MvNormalMeanCovariance },
    on   => :out_mean_covariance,
    messages => (m_out::MvNormalMeanCovariance, m_mean::Dirac, m_covariance::Dirac),
    marginals => Nothing,
    meta => Nothing,
    begin
        q_out = m_out * as_message(MvNormalMeanCovariance(mean(m_mean), mean(m_covariance)))
        return FactorizedMarginal(q_out, m_mean, m_covariance)
    end
)

@marginalrule(
    form => Type{ <: MvNormalMeanCovariance },
    on   => :out_mean_covariance,
    messages => (m_out::Dirac, m_mean::MvNormalMeanCovariance, m_covariance::Dirac),
    marginals => Nothing,
    meta => Nothing,
    begin
        q_mean = m_mean * as_message(MvNormalMeanCovariance(mean(m_out), mean(m_covariance)))
        return FactorizedMarginal(m_out, q_mean, m_covariance)
    end
)

@marginalrule(
    form => Type{ <: MvNormalMeanCovariance },
    on   => :out_mean,
    messages => (m_out::MvNormalMeanCovariance, m_mean::MvNormalMeanCovariance),
    marginals => (q_covariance::Dirac, ),
    meta => Nothing,
    begin
        W_y  = inv(cov(m_out))
        xi_y = W_y * mean(m_out)

        W_m  = inv(cov(m_mean))
        xi_m = W_m * mean(m_mean)

        W_bar = inv(mean(q_covariance))
        
        xi = [ xi_y; xi_m ]
        W  = PDMat(Matrix(Hermitian([ W_y+W_bar -W_bar; -W_bar W_m+W_bar ])))
        
        c = inv(W)
        m = c * xi
        
        return MvNormalMeanCovariance(m, c)
    end
)