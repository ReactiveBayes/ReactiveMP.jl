@marginalrule(
    formtype  => NormalMeanVariance,
    on        => :out_μ_v,
    messages  => (m_out::NormalMeanVariance{T}, m_μ::Dirac{T}, m_v::Dirac{T}) where { T <: Real },
    marginals => Nothing,
    meta      => Nothing,
    begin 
        return (out = prod(ProdPreserveParametrisation(), NormalMeanVariance(mean(m_μ), mean(m_v)), m_out), μ = m_μ, v = m_v)
    end
)

@marginalrule(
    formtype  => NormalMeanVariance,
    on        => :out_μ_v,
    messages  => (m_out::Dirac{T}, m_μ::NormalMeanVariance{T}, m_v::Dirac{T}) where { T <: Real },
    marginals => Nothing,
    meta      => Nothing,
    begin 
        return (out = m_out, μ = prod(ProdPreserveParametrisation(), m_μ, NormalMeanVariance(mean(m_out), mean(m_v))), v = m_v)
    end
)