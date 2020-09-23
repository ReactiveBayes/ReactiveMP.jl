@rule(
    form        => Type{ <: NormalMeanVariance }, 
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_mean::Dirac{T}, m_variance::Dirac{T}) where { T <: Real },
    marginals   => Nothing,
    meta        => Nothing,
    begin 
        return NormalMeanVariance(mean(m_mean), mean(m_variance))
    end
)

@rule(
    form        => Type{ <: NormalMeanVariance }, 
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_mean::NormalMeanVariance{T}, m_variance::Dirac{T}) where { T <: Real },
    marginals   => Nothing,
    meta        => Nothing,
    begin 
        return NormalMeanVariance(mean(m_mean), var(m_mean) + mean(m_variance))
    end
)

@rule(
    form        => Type{ <: NormalMeanVariance }, 
    on          => :out,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_mean::Any, q_variance::Any),
    meta        => Nothing,
    begin 
        return NormalMeanVariance(mean(q_mean), mean(q_variance))
    end
)