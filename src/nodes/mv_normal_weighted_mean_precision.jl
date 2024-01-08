import StatsFuns: log2π

@node MvNormalWeightedMeanPrecision Stochastic [out, (ξ, aliases = [xi, weightedmean]), (Λ, aliases = [invcov, precision])]

@average_energy MvNormalWeightedMeanPrecision (q_out::Any, q_ξ::Any, q_Λ::Any) = begin
    marginals = (Marginal(q_out, false, false, nothing), Marginal(q_ξ, false, false, nothing), Marginal(q_Λ, false, false, nothing))
    score(AverageEnergy(), MvNormalMeanPrecision, Val{(:out, :μ, :Λ)}(), marginals, nothing)
end

@average_energy MvNormalWeightedMeanPrecision (q_out_ξ::Any, q_Λ::Any) = begin
    marginals = (Marginal(q_out_ξ, false, false, nothing), Marginal(q_Λ, false, false, nothing))
    score(AverageEnergy(), MvNormalMeanPrecision, Val{(:out_μ, :Λ)}(), marginals, nothing)
end
