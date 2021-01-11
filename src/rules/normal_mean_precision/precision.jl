export rule

@rule NormalMeanPrecision(:τ, Marginalisation) (q_out::Any, q_μ::Any) = begin
    diff = mean(q_out) - mean(q_μ)
    return Gamma(1.5, 2.0 / (var(q_out) + var(q_μ) + abs2(diff)))
end

@rule NormalMeanPrecision(:τ, Marginalisation) (q_out_μ::Any, ) = begin
    m, V = mean(q_out_μ), cov(q_out_μ)
    return Gamma(1.5, 2.0 / (V[1,1] - V[1,2] - V[2,1] + V[2,2] + abs2(m[1] - m[2])))
end