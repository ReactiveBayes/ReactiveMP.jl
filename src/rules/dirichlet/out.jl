@rule(
    formtype    => Dirichlet,
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_a::Dirac{ <: AbstractVector }, ),
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Dirichlet(mean(m_a))
    end
)