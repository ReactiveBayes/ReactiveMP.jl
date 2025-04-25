module ReactiveMPGenUTExt

using ReactiveMP, ExponentialFamily, Distributions, BayesBase, Random, LinearAlgebra, FastCholesky, GeneralizedUnscented

include("rules/in.jl")
include("rules/out.jl")
include("rules/marginals.jl")

# This will enable the extension and make `GenUnscented` compatible with delta nodes 
ReactiveMP.check_delta_node_compatibility(::GenUnscented) = Val(true)
ReactiveMP.deltafn_rule_layout(::DeltaFnNode, ::GenUnscented, inverse::Nothing) = ReactiveMP.DeltaFnDefaultRuleLayout()

end
