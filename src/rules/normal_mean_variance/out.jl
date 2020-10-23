@rule(
    form        => Type{ <: NormalMeanVariance }, 
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_μ::Dirac{T}, m_v::Dirac{T}) where { T <: Real },
    marginals   => Nothing,
    meta        => Nothing,
    begin 
        return NormalMeanVariance(mean(m_μ), mean(m_v))
    end
)

@rule(
    form        => Type{ <: NormalMeanVariance }, 
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_μ::NormalMeanVariance{T}, m_v::Dirac{T}) where { T <: Real },
    marginals   => Nothing,
    meta        => Nothing,
    begin 
        return NormalMeanVariance(mean(m_μ), var(m_μ) + mean(m_v))
    end
)

@rule(
    form        => Type{ <: NormalMeanVariance }, 
    on          => :out,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_μ::Any, q_v::Any),
    meta        => Nothing,
    begin 
        return NormalMeanVariance(mean(q_μ), mean(q_v))
    end
)