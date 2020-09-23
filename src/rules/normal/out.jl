@rule(
    form        => Type{ <: Normal },
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_mean::Dirac{T}, m_std::Dirac{T}) where { T <: Real },
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return Normal(mean(m_mean), mean(m_std))
    end
)