


function score(::AverageEnergy, ::Type{ <: Wishart }, marginals::Tuple{Marginal, Marginal, Marginal}, ::Nothing)
    
    marg_out = marginals[1]
    marg_v   = marginals[3]
    marg_nu  = marginals[2]

    d = size(getdata(marg_out))[1]

    0.5*mean(marg_nu)*logdet(mean(marg_v)) +
    0.5*mean(marg_nu)*d*log(2) +
    0.25*d*(d - 1.0)*log(pi) +
    sum([labsgamma(0.5*(mean(marg_nu) + 1.0 - i)) for i=1:d]) -
    0.5*(mean(marg_nu) - d - 1.0)*logdet(mean(marg_out)) +
    0.5*tr(inversemean(marg_v)*mean(marg_out))
end


# function averageEnergy(::Type{Wishart}, marg_out::ProbabilityDistribution{MatrixVariate}, marg_v::ProbabilityDistribution{MatrixVariate}, marg_nu::ProbabilityDistribution{Univariate, PointMass})
#     d = dims(marg_out)[1]
#     0.5*marg_nu.params[:m]*unsafeDetLogMean(marg_v) +
#     0.5*marg_nu.params[:m]*d*log(2) +
#     0.25*d*(d - 1.0)*log(pi) +
#     sum([labsgamma(0.5*(marg_nu.params[:m] + 1.0 - i)) for i=1:d]) -
#     0.5*(marg_nu.params[:m] - d - 1.0)*unsafeDetLogMean(marg_out) +
#     0.5*tr(unsafeInverseMean(marg_v)*unsafeMean(marg_out))
# end