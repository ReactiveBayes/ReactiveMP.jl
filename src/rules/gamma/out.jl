@rule(
    form        => Type{ <: Gamma },
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_α::Dirac{T}, m_θ::Dirac{T}) where T,
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Gamma(mean(m_α), mean(m_θ))
    end
)