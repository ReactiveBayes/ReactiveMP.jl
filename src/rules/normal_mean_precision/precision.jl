export rule

@rule NormalMeanPrecision(:τ, Marginalisation) (q_out::Any, q_μ::Any) = begin
    θ = 2 / (var(q_out) + var(q_μ) + abs2(mean(q_out) - mean(q_μ)))
    α = convert(typeof(θ), 1.5)
    return Gamma(α, θ)
end

@rule NormalMeanPrecision(:τ, Marginalisation) (q_out_μ::Any, ) = begin
    m, V = mean_cov(q_out_μ)
    θ = 2 / (V[1,1] - V[1,2] - V[2,1] + V[2,2] + abs2(m[1] - m[2]))
    α = convert(typeof(θ), 1.5)
    return Gamma(α, θ)
end