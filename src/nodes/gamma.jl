export make_node, rule

function make_node(::Type{ <: Gamma }; factorisation = ((1, 2, 3), ))
    return FactorNode(Gamma, Stochastic, (:out, :α, :θ), factorisation, nothing)
end

function make_node(::Type{ <: Gamma }, out::AbstractVariable, α::AbstractVariable, θ::AbstractVariable; factorisation = ((1, 2, 3), ))
    node = make_node(Gamma, factorisation = factorisation)
    connect!(node, :out, out)
    connect!(node, :α, α)
    connect!(node, :θ, θ)
    return node
end

@average_energy(
    form      => Type{ <: Gamma },
    marginals => (q_out::Any, q_α::Any, q_θ::Any),
    meta      => Nothing,
    begin
        return labsgamma(mean(q_α)) + mean(q_α) * log(mean(q_θ)) - (mean(q_α) - 1.0) * log(mean(q_out)) + inv(mean(q_θ)) * mean(q_out)
    end
)
