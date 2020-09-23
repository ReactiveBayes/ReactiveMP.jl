@rule(
    form        => Type{ <: NormalMeanVariance }, 
    on          => :mean,
    vconstraint => Marginalisation,
    messages    => (m_out::Dirac{T}, m_variance::Dirac{T}) where { T <: Real },
    marginals   => Nothing,
    meta        => Nothing,
    begin 
        return NormalMeanVariance(mean(m_out), mean(m_variance))
    end
)

@rule(
    form        => Type{ <: NormalMeanVariance }, 
    on          => :mean,
    vconstraint => Marginalisation,
    messages    => (m_out::NormalMeanVariance{T}, m_variance::Dirac{T}) where { T <: Real },
    marginals   => Nothing,
    meta        => Nothing,
    begin 
        return NormalMeanVariance(mean(m_out), var(m_out) + mean(m_variance))
    end
)

@rule(
    form        => Type{ <: NormalMeanVariance }, 
    on          => :mean,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_out::Any, q_variance::Any),
    meta        => Nothing,
    begin 
        return NormalMeanVariance(mean(q_out), mean(q_variance))
    end
)