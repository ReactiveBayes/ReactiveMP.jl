
import PDMats: logdet

function score(::DifferentialEntropy, marginal::Marginal{ <: MvNormalMeanCovariance })
    dims = ndims(marginal)
    return 0.5 * logdet(cov(marginal)) + (dims / 2.0) * log(2pi) + (dims / 2.0) 
 end