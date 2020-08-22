
function score(::DifferentialEntropy, marginal::Marginal{ <: NormalMeanPrecision })
    return 0.5 * log(var(marginal)) + 0.5 * log(2 * π) + 0.5
end
