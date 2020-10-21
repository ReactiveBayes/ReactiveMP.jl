@rule(
    form        => Type{ <: MvNormalMeanPrecision },
    on          => :μ,
    vconstraint => Marginalisation,
    messages    => (m_out::Dirac, m_Λ::Dirac),
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return MvNormalMeanPrecision(mean(m_out), mean(m_Λ))
    end
)

@rule(
    form        => Type{ <: MvNormalMeanPrecision },
    on          => :μ,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_out::Any, q_Λ::Any),
    meta        => Nothing,
    begin
        return MvNormalMeanPrecision(mean(q_out), mean(q_Λ))
    end
)

@rule(
    form        => Type{ <: MvNormalMeanPrecision },
    on          => :μ,
    vconstraint => Marginalisation,
    messages    => (m_out::MvNormalMeanPrecision, ),
    marginals   => (q_Λ::Any, ),
    meta        => Nothing,
    begin
        return MvNormalMeanPrecision(mean(m_out), Matrix(Hermitian(inv(cov(m_out) + inversemean(q_Λ)))))
    end
)
