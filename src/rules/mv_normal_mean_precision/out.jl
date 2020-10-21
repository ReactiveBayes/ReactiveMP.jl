@rule(
    form        => Type{ <: MvNormalMeanPrecision },
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_μ::Dirac, m_Λ::Dirac),
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return MvNormalMeanPrecision(mean(m_μ), mean(m_Λ))
    end
)

@rule(
    form        => Type{ <: MvNormalMeanPrecision },
    on          => :out,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_μ::Any, q_Λ::Any),
    meta        => Nothing,
    begin
        return MvNormalMeanPrecision(mean(q_μ), mean(q_Λ))
    end
)

@rule(
    form        => Type{ <: MvNormalMeanPrecision },
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_μ::MvNormalMeanPrecision, ),
    marginals   => (q_Λ::Any, ),
    meta        => Nothing,
    begin
        # @show inv(mean(q_Λ))
        return MvNormalMeanPrecision(mean(m_μ), Matrix(Hermitian(inv(Matrix(Hermitian(cov(m_μ) + inversemean(q_Λ)))))))
    end
)