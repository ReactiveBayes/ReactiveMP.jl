@marginalrule(
    form        => Type{ <: NormalMeanPrecision },
    on          => :out_mean_precision,
    messages    => (m_out::NormalMeanPrecision{T}, m_mean::Dirac{T}, m_precision::Dirac{T}) where { T <: Real },
    marginals   => Nothing,
    meta        => Nothing,
    begin
        q_out = as_message(NormalMeanPrecision(mean(m_mean), mean(m_precision))) * m_out
        return FactorizedMarginal(q_out, m_mean, m_precision)
    end
)