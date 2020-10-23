
import Distributions: Distribution, entropy

function score(::DifferentialEntropy, marginal::Marginal{ <: Tuple })
    return mapreduce((d) -> score(DifferentialEntropy(), as_marginal(d)), +, getdata(marginal))
end

function score(::DifferentialEntropy, marginal::Marginal{ <: Distribution })
    return entropy(marginal)
end