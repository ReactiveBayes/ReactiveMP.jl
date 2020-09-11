export make_node, rule

import Distributions: Normal

function make_node(::Type{ <: Normal })
    return FactorNode(Normal, Stochastic, (:out, :mean, :std), ((1, 2, 3), ), nothing)
end

function make_node(::Type{ <: Normal }, out, mean, std)
    node = make_node(Normal)
    connect!(node, :out, out)
    connect!(node, :mean, mean)
    connect!(node, :std, std)
    return node
end

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
