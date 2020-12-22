export make_node, conjugate_type, score

import StatsFuns: log2π

@node(
    formtype   => NormalMeanVariance,
    sdtype     => Stochastic,
    interfaces => [
        out,
        (μ, aliases = [ mean ]),
        (v, aliases = [ var ])
    ]
)

conjugate_type(::Type{ <: NormalMeanVariance }, ::Type{ Val{ :out } }) = NormalMeanVariance
conjugate_type(::Type{ <: NormalMeanVariance }, ::Type{ Val{ :μ } })   = NormalMeanVariance
conjugate_type(::Type{ <: NormalMeanVariance }, ::Type{ Val{ :v } })   = InverseGamma

@average_energy NormalMeanVariance (q_out::Any, q_μ::Any, q_v::Any) = begin
    μ_mean, μ_var     = mean(q_μ), var(q_μ)
    out_mean, out_var = mean(q_out), var(q_out)
    return 0.5 * (log2π + logmean(q_v) + inv(mean(q_v)) * (μ_var + out_var + abs2(μ_mean - out_mean)))
end