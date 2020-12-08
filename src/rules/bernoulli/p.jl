
@rule(
    formtype    => Bernoulli,
    on          => :p,
    vconstraint => Marginalisation,
    messages    => (m_out::Dirac{T}, ) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Beta(one(T) + mean(m_out), 2 * one(T) - mean(m_out))
    end
)

@rule(
    formtype    => Bernoulli,
    on          => :p,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_out::Any, ),
    meta        => Nothing,
    begin
        return Beta(1.0 + mean(q_out), 2.0 - mean(q_out))
    end
)