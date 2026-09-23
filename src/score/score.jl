export score, DifferentialEntropy

function score end

##

struct DifferentialEntropy end

struct KLDivergence end

## Differential entropy function helpers

# A `FactorizedCluster`'s entropy is the sum over its blocks, which BayesBase's
# `FactorizedJoint` gives; v6 needed a method of its own for its NamedTuple joints.
score(::DifferentialEntropy, marginal::Marginal) = entropy(marginal)

## Kl KlDivergence

score(::KLDivergence, marginal::Marginal, p::Distribution) =
    Distributions.kldivergence(getdata(marginal), p)
