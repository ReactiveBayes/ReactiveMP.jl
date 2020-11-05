@rule(
    formtype    => NormalMeanVariance,
    on          => :μ,
    vconstraint => Marginalisation,
    messages    => (m_out::Dirac{T}, m_v::Dirac{T}) where { T <: Real },
    marginals   => Nothing,
    meta        => Nothing,
    begin 
        return NormalMeanVariance(mean(m_out), mean(m_v))
    end
)

@rule(
    formtype    => NormalMeanVariance,
    on          => :μ,
    vconstraint => Marginalisation,
    messages    => (m_out::NormalMeanVariance{T}, m_v::Dirac{T}) where { T <: Real },
    marginals   => Nothing,
    meta        => Nothing,
    begin 
        return NormalMeanVariance(mean(m_out), var(m_out) + mean(m_v))
    end
)

@rule(
    formtype    => NormalMeanVariance,
    on          => :μ,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_out::Any, q_v::Any),
    meta        => Nothing,
    begin 
        return NormalMeanVariance(mean(q_out), mean(q_v))
    end
)