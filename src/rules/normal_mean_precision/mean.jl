@rule(
    formtype    => NormalMeanPrecision,
    on          => :μ,
    vconstraint => Marginalisation,
    messages    => (m_out::Dirac{T}, m_τ::Dirac{T}) where { T <: Real },
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return NormalMeanPrecision(mean(m_out), mean(m_τ))
    end
)

@rule(
    formtype    => NormalMeanPrecision,
    on          => :μ,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_out::Any, q_τ::Any),
    meta        => Nothing,
    begin
        return NormalMeanPrecision(mean(q_out), mean(q_τ))
    end
)

@rule(
    formtype    => NormalMeanPrecision,
    on          => :μ,
    vconstraint => Marginalisation,
    messages    => (m_out::NormalMeanPrecision, ),
    marginals   => (q_τ::Any, ),
    meta        => Nothing,
    begin
        return NormalMeanPrecision(mean(m_out), cholinv( cov(m_out) + cholinv(mean(q_τ)) ))
    end
)