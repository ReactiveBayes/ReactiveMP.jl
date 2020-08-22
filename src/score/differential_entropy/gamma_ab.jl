function score(::DifferentialEntropy, marginal::Marginal{ <: GammaAB })
    dist = getdata(marginal)
    return labsgamma(dist.a) - (dist.a - 1.0) * digamma(dist.a) - log(dist.b) + dist.a
end