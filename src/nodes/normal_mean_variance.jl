export make_node, rule


function make_node(::Type{ <: NormalMeanVariance }; factorisation = ((1, 2, 3), ))
    return FactorNode(NormalMeanVariance, Stochastic, (:out, :mean, :variance), factorisation, nothing)
end

function make_node(::Type{ <: NormalMeanVariance }, out, mean, variance; factorisation = ((1, 2, 3), ))
    node = make_node(NormalMeanVariance, factorisation = factorisation)
    connect!(node, :out, out)
    connect!(node, :mean, mean)
    connect!(node, :variance, variance)
    return node
end

## rules

@rule(
    form        => Type{ <: NormalMeanVariance }, 
    on          => :mean,
    vconstraint => Marginalisation,
    messages    => (m_out::Dirac{T}, m_variance::Dirac{T}) where { T <: Real },
    marginals   => Nothing,
    meta        => Nothing,
    begin 
        return NormalMeanVariance(mean(m_out), mean(m_variance))
    end
)

@rule(
    form        => Type{ <: NormalMeanVariance }, 
    on          => :mean,
    vconstraint => Marginalisation,
    messages    => (m_out::NormalMeanVariance{T}, m_variance::Dirac{T}) where { T <: Real },
    marginals   => Nothing,
    meta        => Nothing,
    begin 
        return NormalMeanVariance(mean(m_out), var(m_out) + mean(m_variance))
    end
)

@rule(
    form        => Type{ <: NormalMeanVariance }, 
    on          => :mean,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_out::Any, q_variance::Any),
    meta        => Nothing,
    begin 
        return NormalMeanVariance(mean(q_out), mean(q_variance))
    end
)

@rule(
    form        => Type{ <: NormalMeanVariance }, 
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_mean::Dirac{T}, m_variance::Dirac{T}) where { T <: Real },
    marginals   => Nothing,
    meta        => Nothing,
    begin 
        return NormalMeanVariance(mean(m_mean), mean(m_variance))
    end
)

@rule(
    form        => Type{ <: NormalMeanVariance }, 
    on          => :out,
    vconstraint => Marginalisation,
    messages    => (m_mean::NormalMeanVariance{T}, m_variance::Dirac{T}) where { T <: Real },
    marginals   => Nothing,
    meta        => Nothing,
    begin 
        return NormalMeanVariance(mean(m_mean), var(m_mean) + mean(m_variance))
    end
)

@rule(
    form        => Type{ <: NormalMeanVariance }, 
    on          => :out,
    vconstraint => Marginalisation,
    messages    => Nothing,
    marginals   => (q_mean::Any, q_variance::Any),
    meta        => Nothing,
    begin 
        return NormalMeanVariance(mean(q_mean), mean(q_variance))
    end
)

@marginalrule(
    form      => Type{ <: NormalMeanVariance },
    on        => :out_mean_variance,
    messages  => (m_out::NormalMeanVariance{T}, m_mean::Dirac{T}, m_variance::Dirac{T}) where { T <: Real },
    marginals => Nothing,
    meta      => Nothing,
    begin 
        q_out = Message(NormalMeanVariance(mean(m_mean), mean(m_variance))) * m_out
        return (getdata(q_out), getdata(m_mean), getdata(m_variance))
    end
)

@marginalrule(
    form      => Type{ <: NormalMeanVariance },
    on        => :out_mean_variance,
    messages  => (m_out::Dirac{T}, m_mean::NormalMeanVariance{T}, m_variance::Dirac{T}) where { T <: Real },
    marginals => Nothing,
    meta      => Nothing,
    begin 
        q_mean = Message(NormalMeanVariance(mean(m_out), mean(m_variance))) * m_mean
        return (getdata(m_out), getdata(q_mean), getdata(m_variance))
    end
)