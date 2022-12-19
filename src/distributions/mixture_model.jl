export MixtureModel

import Distributions: MixtureModel
import Base: show

using Distributions: component_type

function MixtureModel(components::Vector, prior::Bernoulli)
    @assert length(components) == 2 "A mixture model can only be specified with a Bernoulli distribution, if there are 2 components."
    prior_cat = Categorical([prior.p, 1 - prior.p])
    return MixtureModel(components, prior_cat)
end

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

prod(::ProdAnalytical, left::Any, right::MixtureModel) = prod(ProdAnalytical(), right, left)

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

prod(::AddonProdLogScale, new_dist::MixtureModel, left_dist::Any, right_dist::MixtureModel) = prod(AddonProdLogScale(), new_dist, right_dist, left_dist)

# improved printing in nested statements

# commented out by bvdmitri: it overrides the default printing:
# WARNING: Method definition show(IO, Distributions.MixtureModel{VF, VS, C, CT} where CT<:(Distributions.DiscreteNonParametric{Int64, P, Base.OneTo{Int64}, Ps} where Ps<:AbstractArray{P, 1} where P<:Real) where C<:(Distributions.Distribution{F, S} where S<:Distributions.ValueSupport where F<:Distributions.VariateForm) where VS<:Distributions.ValueSupport where VF<:Distributions.VariateForm) in module Distributions at /Users/bvdmitri/.julia/packages/Distributions/t2DAm/src/mixtures/mixturemodel.jl:257 overwritten in module ReactiveMP at /Users/bvdmitri/.julia/dev/ReactiveMP.jl/src/distributions/mixture_model.jl:54.
# ** incremental compilation may be fatally broken for this module **

# function show(io::IO, d::MixtureModel)
#     indent = get(io, :indent, 0)
#     K = ncomponents(d)
#     pr = probs(d)
#     println(io, "\n", ' '^indent, "MixtureModel{$(component_type(d))}(K = $K)")
#     Ks = min(K, 8)
#     for i = 1:Ks
#         print(io, ' '^(indent+4), "components[", string(i), "] (prior = ", string(round.(pr[i]; digits=4)), ")")
#         println(io, component(d, i))
#     end
#     if Ks < K
#         println(io, "The rest are omitted ...")
#     end
# end
