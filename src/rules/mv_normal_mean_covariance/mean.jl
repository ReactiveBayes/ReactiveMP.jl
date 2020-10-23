@rule(
    form        => Type{ <: MvNormalMeanCovariance },
    on          => :μ,
    vconstraint => Marginalisation,
    messages    => (m_out::Dirac, m_Σ::Dirac),
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return MvNormalMeanCovariance(mean(m_out), mean(m_Σ))
    end
)

@rule(
    form        => Type{ <: MvNormalMeanCovariance },
    on          => :μ,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_out::Any, q_Σ::Any),
    meta        => Nothing,
    begin
        return MvNormalMeanCovariance(mean(q_out), mean(q_Σ))
    end
)

@rule(
    form        => Type{ <: MvNormalMeanCovariance },
    on          => :μ,
    vconstraint => Marginalisation,
    messages    => (m_out::MvNormalMeanCovariance, ),
    marginals   => (q_Σ::Any, ),
    meta        => Nothing,
    begin
        return MvNormalMeanCovariance(mean(m_out), cov(m_out) + mean(q_Σ))
    end
)
