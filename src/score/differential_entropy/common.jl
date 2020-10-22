
import Distributions: Distribution, entropy

function score(::DifferentialEntropy, fmarginal::Marginal{ <: FactorizedMarginal })
    return mapreduce((d) -> score(DifferentialEntropy(), as_marginal(d)), +, fmarginal |> getdata |> getfactors)
end

function score(::DifferentialEntropy, marginal::Marginal{ <: Distribution })
    return entropy(marginal)
end