
function score(::DifferentialEntropy, marginal::Marginal{ <: Dirac }) where { T <: Real }
    return ∞
end