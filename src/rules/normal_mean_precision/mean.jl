@rule(
    form        => Type{ <: NormalMeanPrecision },
    on          => :mean,
    vconstraint => Marginalisation,
    messages    => (m_out::Dirac{T}, m_precision::Dirac{T}) where { T <: Real },
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return NormalMeanPrecision(mean(m_out), mean(m_precision))
    end
)

@rule(
    form        => Type{ <: NormalMeanPrecision },
    on          => :mean,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_out::Any, q_precision::Any),
    meta        => Nothing,
    begin
        return NormalMeanPrecision(mean(q_out), mean(q_precision))
    end
)