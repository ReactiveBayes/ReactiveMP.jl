@marginalrule(
    form      => Type{ <: NormalMeanVariance },
    on        => :out_mean_variance,
    messages  => (m_out::NormalMeanVariance{T}, m_mean::Dirac{T}, m_variance::Dirac{T}) where { T <: Real },
    marginals => Nothing,
    meta      => Nothing,
    begin 
        return (prod(ProdPreserveParametrisation(), NormalMeanVariance(mean(m_mean), mean(m_variance)), m_out), m_mean, m_variance)
    end
)

@marginalrule(
    form      => Type{ <: NormalMeanVariance },
    on        => :out_mean_variance,
    messages  => (m_out::Dirac{T}, m_mean::NormalMeanVariance{T}, m_variance::Dirac{T}) where { T <: Real },
    marginals => Nothing,
    meta      => Nothing,
    begin 
        return (m_out, prod(ProdPreserveParametrisation(), m_mean, NormalMeanVariance(mean(m_out), mean(m_variance))), m_variance)
    end
)