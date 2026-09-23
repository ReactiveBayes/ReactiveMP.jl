"""
    StandardMessagePassingRules

The message passing rules for the standard nodes: distributions, arithmetic, logic and the
mixtures, written with `MessagePassingRulesBase`. A distribution node runs under `BP` by
default, and its rules combine messages and marginals as v6's did: the engine's default
dependency scheme gives each rule the messages inside its own cluster and the marginals of
the other clusters. The mixtures run under `VMP`, with declared dependencies.
"""
module StandardMessagePassingRules

using MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions
using MessagePassingRulesBase: annotate!, BP, VMP
using StatsFuns: log2π

# The nodes this package declares, for types other packages own.
const NODES = [NormalMeanVariance, NormalMeanPrecision]

include("nodes/normal_mean_variance.jl")
include("rules/normal_mean_variance/out.jl")
include("rules/normal_mean_variance/mean.jl")
include("rules/normal_mean_variance/marginals.jl")

include("nodes/normal_mean_precision.jl")
include("rules/normal_mean_precision/out.jl")
include("rules/normal_mean_precision/mean.jl")
include("rules/normal_mean_precision/precision.jl")
include("rules/normal_mean_precision/marginals.jl")

end
