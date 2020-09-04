
function score(::DifferentialEntropy, marginal::Marginal{ <: Tuple })
    return mapreduce((d) -> score(DifferentialEntropy(), as_marginal(d)), +, getdata(marginal), init = 0.0)
end