@marginalrule(
    form      => Type{ <: NormalMeanVariance },
    on        => :out_mean_variance,
    messages  => (m_out::NormalMeanVariance{T}, m_mean::Dirac{T}, m_variance::Dirac{T}) where { T <: Real },
    marginals => Nothing,
    meta      => Nothing,
    begin 
        q_out = Message(NormalMeanVariance(mean(m_mean), mean(m_variance))) * m_out
        return FactorizedMarginal(q_out, m_mean, m_variance)
    end
)

@marginalrule(
    form      => Type{ <: NormalMeanVariance },
    on        => :out_mean_variance,
    messages  => (m_out::Dirac{T}, m_mean::NormalMeanVariance{T}, m_variance::Dirac{T}) where { T <: Real },
    marginals => Nothing,
    meta      => Nothing,
    begin 
        q_mean = Message(NormalMeanVariance(mean(m_out), mean(m_variance))) * m_mean
        return FactorizedMarginal(m_out, q_mean, m_variance)
    end
)