
function score(::DifferentialEntropy, marginal::Marginal{ <: Dirac{T} }) where { T <: Real }
    return 0.0 # TODO like complex numbers
end