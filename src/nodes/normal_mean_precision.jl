export make_node, conjugate_type, score

import StatsFuns: log2π

@node(
    formtype   => NormalMeanPrecision,
    sdtype     => Stochastic,
    interfaces => [
        out,
        (μ, aliases = [ mean ]),
        (τ, aliases = [ invcov, precision ])
    ]
)

conjugate_type(::Type{ <: NormalMeanPrecision }, ::Type{ Val{ :out } }) = NormalMeanPrecision
conjugate_type(::Type{ <: NormalMeanPrecision }, ::Type{ Val{ :μ } })   = NormalMeanPrecision
conjugate_type(::Type{ <: NormalMeanPrecision }, ::Type{ Val{ :τ } })   = Gamma

@average_energy NormalMeanPrecision (q_out::Any, q_μ::Any, q_τ::Any) = begin
    μ_mean, μ_var     = mean(q_μ), var(q_μ)
    out_mean, out_var = mean(q_out), var(q_out)
    return 0.5 * (log2π - log(mean(q_τ)) + mean(q_τ) * (μ_var + out_var + abs2(μ_mean - out_mean)))
end