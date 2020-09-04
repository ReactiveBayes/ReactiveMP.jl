
function score(::DifferentialEntropy, marginal::Marginal{ <: NormalMeanVariance })
    return 0.5 * log(var(marginal)) + 0.5 * log(2pi) + 0.5
end