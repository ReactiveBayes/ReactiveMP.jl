export MixtureModel

import Distributions: MixtureModel

function prod(::ProdAnalytical, left::MixtureModel, right::Any)
    
    # get prior weights and components
    w = probvec(left.prior)
    dists = left.components

    # get new distributions
    dists_new = map(dist -> prod(ProdAnalytical(), dist, right), dists)

    # get scales
    logscales = map((dist, dist_new) -> prod(AddonProdLogScale(), dist_new, dist, right), dists, dists_new)

    # compute updated weights
    logscales_new = log.(w) + logscales

    # return mixture distributions
    return MixtureModel(dists_new, Categorical(softmax(logscales_new)))

end

function prod(::AddonProdLogScale, new_dist::MixtureModel, left_dist::MixtureModel, right_dist::Any)

    # get prior weights and components
    w = probvec(left_dist.prior)
    dists = left_dist.components

    # get new distributions
    dists_new = map(dist -> prod(ProdAnalytical(), dist, right_dist), dists)

    # get scales
    logscales = map((dist, dist_new) -> prod(AddonProdLogScale(), dist_new, dist, right_dist), dists, dists_new)

    # compute updated weights
    logscales_new = log.(w) + logscales

    return logsumexp(logscales_new)

end