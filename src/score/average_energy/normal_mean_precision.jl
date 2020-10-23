import StatsFuns: log2π

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