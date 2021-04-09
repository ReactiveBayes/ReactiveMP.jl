import StatsFuns: logπ

@node ComplexNormal Stochastic [ out, (μ, aliases = [ mean ]), (Γ, aliases = [ cov, covariance, variance ]), (C, aliases = [ relation ]) ]

@average_energy ComplexNormal (q_out::Any, q_μ::Any, q_Γ::Any, q_C::Any) = begin
    μ_mean, μ_var     = mean(q_μ), real(var(q_μ))
    out_mean, out_var = mean(q_out), real(var(q_out))
    Γ_mean = real(mean(q_Γ))
    return logπ + log(Γ_mean) + 1/Γ_mean * (μ_var + out_var + abs2(μ_mean - out_mean))
end