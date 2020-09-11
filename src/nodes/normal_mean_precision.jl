export make_node, rule


function make_node(::Type{ <: NormalMeanPrecision }; factorisation = ((1, 2, 3), ))
    return FactorNode(NormalMeanPrecision, Stochastic, (:out, :mean, :precision), factorisation, nothing)
end

function make_node(::Type{ <: NormalMeanPrecision }, out, mean, precision; factorisation = ((1, 2, 3), ))
    node = make_node(NormalMeanPrecision, factorisation = factorisation)
    connect!(node, :out, out)
    connect!(node, :mean, mean)
    connect!(node, :precision, precision)
    return node
end

## rules

@rule(
    form        => Type{ <: NormalMeanPrecision },
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_mean::Dirac{T}, m_precision::Dirac{T}) where { T <: Real },
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return NormalMeanPrecision(mean(m_mean), mean(m_precision))
    end
)

@rule(
    form        => Type{ <: NormalMeanPrecision },
    on          => :out,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_mean::Any, q_precision::Any),
    meta        => Nothing,
    begin
        return NormalMeanPrecision(mean(q_mean), mean(q_precision))
    end
)

@rule(
    form        => Type{ <: NormalMeanPrecision },
    on          => :mean,
    vconstraint => Marginalisation,
    messages    => (m_out::Dirac{T}, m_precision::Dirac{T}) where { T <: Real },
    marginals   => Nothing,
    meta        => Nothing,
    begin
        return NormalMeanPrecision(mean(m_out), mean(m_precision))
    end
)

@rule(
    form        => Type{ <: NormalMeanPrecision },
    on          => :mean,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_out::Any, q_precision::Any),
    meta        => Nothing,
    begin
        return NormalMeanPrecision(mean(q_out), mean(q_precision))
    end
)

@rule(
    form        => Type{ <: NormalMeanPrecision },
    on          => :precision,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_out::Any, q_mean::Any),
    meta        => Nothing,
    begin
        diff = mean(marginals[1]) - mean(marginals[2])
        return GammaAB(3.0 / 2.0, 1.0 / 2.0 * (var(marginals[2]) + var(marginals[1]) + diff^2))
    end
)

## marginal rules

@marginalrule(
    form        => Type{ <: NormalMeanPrecision },
    on          => :out_mean_precision,
    messages    => (m_out::NormalMeanPrecision{T}, m_mean::Dirac{T}, m_precision::Dirac{T}) where { T <: Real },
    marginals   => Nothing,
    meta        => Nothing,
    begin
        q_out = Message(NormalMeanPrecision(mean(m_mean), mean(m_precision))) * m_out
        return (getdata(q_out), getdata(m_mean), getdata(m_precision))
    end
)