export make_node, rule

import StatsFuns: log2π

function make_node(::Type{ <: NormalMeanVariance }; factorisation = ((1, 2, 3), ))
    return FactorNode(NormalMeanVariance, Stochastic, (:out, :μ, :v), factorisation, nothing)
end

function make_node(::Type{ <: NormalMeanVariance }, out::AbstractVariable, μ::AbstractVariable, v::AbstractVariable; factorisation = ((1, 2, 3), ))
    node = make_node(NormalMeanVariance, factorisation = factorisation)
    connect!(node, :out, out)
    connect!(node, :μ, μ)
    connect!(node, :v, v)
    return node
end

@average_energy(
    form      => Type{ <: NormalMeanVariance },
    marginals => (q_out::Any, q_μ::Any, q_v::Any),
    meta      => Nothing,
    begin
        μ_mean, μ_var     = mean(q_μ), var(q_μ)
        out_mean, out_var = mean(q_out), var(q_out)
        return 0.5 * (log2π + log(mean(q_v)) + inv(mean(q_v)) * (μ_var + out_var + abs2(μ_mean - out_mean)))
    end
)
