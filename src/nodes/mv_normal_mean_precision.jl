export make_node, rule

import StatsFuns: log2π

# function make_node(::Type{ <: MvNormalMeanPrecision }; factorisation = ((1, 2, 3), ))
#     return FactorNode(MvNormalMeanPrecision, Stochastic, (:out, :μ, :Λ), factorisation, nothing)
# end

# function make_node(::Type{ <: MvNormalMeanPrecision }, out::AbstractVariable, μ::AbstractVariable, Λ::AbstractVariable; factorisation = ((1, 2, 3), ))
#     node = make_node(MvNormalMeanPrecision, factorisation = factorisation)
#     connect!(node, :out, out)
#     connect!(node, :μ, μ)
#     connect!(node, :Λ, Λ)
#     return node
# end

@node(
    form       => MvNormalMeanPrecision,
    formtype   => MvNormalMeanPrecision,
    sdtype     => Stochastic,
    interfaces => [
        out,
        (μ, aliases = [ mean ]),
        (Λ, aliases = [ invcov, precision ])
    ]
)

@average_energy(
    form      => Type{ <: MvNormalMeanPrecision },
    marginals => (q_out::Any, q_μ::Any, q_Λ::Any),
    meta      => Nothing,
    begin
        (m_mean, v_mean) = mean(q_μ), cov(q_μ)
        (m_out, v_out)   = mean(q_out), cov(q_out)

        0.5 * (ndims(q_out) * log2π + logdet(cholinv(mean(q_Λ))) + tr(mean(q_Λ)*(v_out + v_mean + (m_out - m_mean)*(m_out - m_mean)')))
    end
)

@average_energy(
    form      => Type{ <: MvNormalMeanPrecision },
    marginals => (q_out_μ::Any, q_Λ::Any),
    meta      => Nothing,
    begin
        (m, V) = mean(q_out_μ), cov(q_out_μ)
        d = Int64(ndims(q_out_μ)/2)
        @views 0.5*(d*log2π + logdet(cholinv(mean(q_Λ))) + tr(mean(q_Λ)*( V[1:d,1:d] - V[1:d,d+1:end] - V[d+1:end,1:d] + V[d+1:end,d+1:end] + (m[1:d] - m[d+1:end])*(m[1:d] - m[d+1:end])' ) ))
    end
)