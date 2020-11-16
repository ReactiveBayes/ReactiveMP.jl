
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