@marginalrule(
    form        => Type{ <: NormalMeanPrecision },
    on          => :out_mean_precision,
    messages    => (m_out::NormalMeanPrecision{T}, m_mean::Dirac{T}, m_precision::Dirac{T}) where { T <: Real },
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return (prod(ProdPreserveParametrisation(), NormalMeanPrecision(mean(m_mean), mean(m_precision)), m_out), m_mean, m_precision)
    end
)

@marginalrule(
    form        => Type{ <: NormalMeanPrecision },
    on          => :out_mean_precision,
    messages    => (m_out::Dirac{T}, m_mean::NormalMeanPrecision{T}, m_precision::Dirac{T}) where { T <: Real },
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return (m_out, prod(ProdPreserveParametrisation(), m_mean, NormalMeanPrecision(mean(m_out), mean(m_precision))), m_precision)
    end
)