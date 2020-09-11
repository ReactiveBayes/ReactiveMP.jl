export make_node, rule

function make_node(::Type{ <: MvNormalMeanCovariance }; factorisation = ((1, 2, 3), ))
    return FactorNode(MvNormalMeanCovariance, Stochastic, (:out, :mean, :covariance), factorisation, nothing)
end

function make_node(::Type{ <: MvNormalMeanCovariance }, out, mean, covariance; factorisation = ((1, 2, 3), ))
    node = make_node(MvNormalMeanCovariance, factorisation = factorisation)
    connect!(node, :out, out)
    connect!(node, :mean, mean)
    connect!(node, :covariance, covariance)
    return node
end

## rules

@rule(
    form        => Type{ <: MvNormalMeanCovariance },
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_mean::Dirac, m_covariance::Dirac),
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return MvNormalMeanCovariance(mean(m_mean), mean(m_covariance))
    end
)

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
    on          => :out,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_mean::Any, q_covariance::Any),
    meta        => Nothing,
    begin
        return MvNormalMeanCovariance(mean(q_mean), mean(q_covariance))
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
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_mean::MvNormalMeanCovariance, ),
    marginals   => (q_covariance::Any, ),
    meta        => Nothing,
    begin
        return MvNormalMeanCovariance(mean(m_mean), cov(m_mean) + mean(q_covariance))
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