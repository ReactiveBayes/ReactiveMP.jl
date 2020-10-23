@marginalrule(
    form      => Type{ <: NormalMeanVariance },
    on        => :out_μ_v,
    messages  => (m_out::NormalMeanVariance{T}, m_μ::Dirac{T}, m_v::Dirac{T}) where { T <: Real },
    marginals => Nothing,
    meta      => Nothing,
    begin 
        return (prod(ProdPreserveParametrisation(), NormalMeanVariance(mean(m_μ), mean(m_v)), m_out), m_μ, m_v)
    end
)

@marginalrule(
    form      => Type{ <: NormalMeanVariance },
    on        => :out_μ_v,
    messages  => (m_out::Dirac{T}, m_μ::NormalMeanVariance{T}, m_v::Dirac{T}) where { T <: Real },
    marginals => Nothing,
    meta      => Nothing,
    begin 
        return (m_out, prod(ProdPreserveParametrisation(), m_μ, NormalMeanVariance(mean(m_out), mean(m_v))), m_v)
    end
)