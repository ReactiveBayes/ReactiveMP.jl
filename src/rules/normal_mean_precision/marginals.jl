@marginalrule(
    form        => Type{ <: NormalMeanPrecision },
    on          => :out_μ_τ,
    messages    => (m_out::NormalMeanPrecision{T}, m_μ::Dirac{T}, m_τ::Dirac{T}) where { T <: Real },
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return (prod(ProdPreserveParametrisation(), NormalMeanPrecision(mean(m_μ), mean(m_τ)), m_out), m_μ, m_τ)
    end
)

@marginalrule(
    form        => Type{ <: NormalMeanPrecision },
    on          => :out_mean_precision,
    messages    => (m_out::Dirac{T}, m_μ::NormalMeanPrecision{T}, m_τ::Dirac{T}) where { T <: Real },
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return (m_out, prod(ProdPreserveParametrisation(), m_μ, NormalMeanPrecision(mean(m_out), mean(m_τ))), m_τ)
    end
)