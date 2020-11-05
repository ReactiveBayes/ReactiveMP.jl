@rule(
    formtype    => MvNormalMeanCovariance,
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_μ::Dirac, m_Σ::Dirac),
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return MvNormalMeanCovariance(mean(m_μ), mean(m_Σ))
    end
)

@rule(
    formtype    => MvNormalMeanCovariance,
    on          => :out,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_μ::Any, q_Σ::Any),
    meta        => Nothing,
    begin
        return MvNormalMeanCovariance(mean(q_μ), mean(q_Σ))
    end
)

@rule(
    formtype    => MvNormalMeanCovariance,
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_μ::MvNormalMeanCovariance, ),
    marginals   => (q_Σ::Any, ),
    meta        => Nothing,
    begin
        return MvNormalMeanCovariance(mean(m_μ), cov(m_μ) + mean(q_Σ))
    end
)