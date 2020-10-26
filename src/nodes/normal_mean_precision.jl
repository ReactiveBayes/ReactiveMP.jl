export make_node, rule

import StatsFuns: log2π

function make_node(::Type{ <: NormalMeanPrecision }; factorisation = ((1, 2, 3), ))
    return FactorNode(NormalMeanPrecision, Stochastic, (:out, :μ, :τ), factorisation, nothing)
end

function make_node(::Type{ <: NormalMeanPrecision }, out, μ, τ; factorisation = ((1, 2, 3), ))
    node = make_node(NormalMeanPrecision, factorisation = factorisation)
    connect!(node, :out, out)
    connect!(node, :μ, μ)
    connect!(node, :τ, τ)
    return node
end

@average_energy(
    form      => Type{ <: NormalMeanPrecision },
    marginals => (q_out::Any, q_μ::Any, q_τ::Any),
    meta      => Nothing,
    begin
        μ_mean, μ_var     = mean(q_μ), var(q_μ)
        out_mean, out_var = mean(q_out), var(q_out)
        return 0.5 * (log2π - log(mean(q_τ)) + mean(q_τ) * (μ_var + out_var + abs2(μ_mean - out_mean)))
    end
)