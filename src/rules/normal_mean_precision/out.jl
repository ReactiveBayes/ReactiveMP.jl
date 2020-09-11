@rule(
    form        => Type{ <: NormalMeanPrecision },
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_mean::Dirac{T}, m_precision::Dirac{T}) where { T <: Real },
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return NormalMeanPrecision(mean(m_mean), mean(m_precision))
    end
)

@rule(
    form        => Type{ <: NormalMeanPrecision },
    on          => :out,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_mean::Any, q_precision::Any),
    meta        => Nothing,
    begin
        return NormalMeanPrecision(mean(q_mean), mean(q_precision))
    end
)