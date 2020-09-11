@rule(
    form        => Type{ <: MvNormalMeanCovariance },
    on          => :mean,
    vconstraint => Marginalisation,
    messages    => (m_out::Dirac, m_covariance::Dirac),
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return MvNormalMeanCovariance(mean(m_out), mean(m_covariance))
    end
)

@rule(
    form        => Type{ <: MvNormalMeanCovariance },
    on          => :mean,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_out::Any, q_covariance::Any),
    meta        => Nothing,
    begin
        return MvNormalMeanCovariance(mean(q_out), mean(q_covariance))
    end
)

@rule(
    form        => Type{ <: MvNormalMeanCovariance },
    on          => :mean,
    vconstraint => Marginalisation,
    messages    => (m_out::MvNormalMeanCovariance, ),
    marginals   => (q_covariance::Any, ),
    meta        => Nothing,
    begin
        return MvNormalMeanCovariance(mean(m_out), cov(m_out) + mean(q_covariance))
    end
)