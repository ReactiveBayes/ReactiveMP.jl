
import Distributions: Distribution, entropy

function score(::DifferentialEntropy, fmarginal::Marginal{ <: FactorizedMarginal })
    return mapreduce((d) -> score(DifferentialEntropy(), as_marginal(d)), +, fmarginal |> getdata |> getfactors, init = InfCountingReal(zero(Float64)))
end

function score(::DifferentialEntropy, marginal::Marginal{ <: Distribution })
    return entropy(marginal)
end