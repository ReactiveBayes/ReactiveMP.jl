@rule(
    form        => Type{ <: GammaAB },
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_a::Dirac{T}, m_b::Dirac{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return GammaAB(mean(m_a), mean(m_b))
    end
)