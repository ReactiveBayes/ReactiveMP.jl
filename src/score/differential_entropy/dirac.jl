
function score(::DifferentialEntropy, ::Marginal{ <: Dirac })
    return ∞
end