@rule(
    formtype    => NormalMeanPrecision,
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_μ::Dirac{T}, m_τ::Dirac{T}) where { T <: Real },
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return NormalMeanPrecision(mean(m_μ), mean(m_τ))
    end
)

@rule(
    formtype    => NormalMeanPrecision,
    on          => :out,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_μ::Any, q_τ::Any),
    meta        => Nothing,
    begin
        return NormalMeanPrecision(mean(q_μ), mean(q_τ))
    end
)