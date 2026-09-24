"""
    FlowMessagePassingRules

The [`Flow`](@ref) node, `out = f(in)` for an invertible flow model `f`, with the models it is
built from: [`FlowModel`](@ref) and its layers, [`InputLayer`](@ref),
[`AdditiveCouplingLayer`](@ref) and [`PermutationLayer`](@ref), the coupling flows
[`PlanarFlow`](@ref) and [`RadialFlow`](@ref), and [`PermutationMatrix`](@ref). Its rules run under
[`FlowApproximation`](@ref), which the model must give.
"""
module FlowMessagePassingRules

using MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions
using MessagePassingRulesBase: AbstractAlgorithm
using MessagePassingRulesApproximations: Linearization, Unscented
import MessagePassingRulesApproximations
using LinearAlgebra
using LinearAlgebra: Adjoint, Transpose
using Random: AbstractRNG, default_rng, shuffle
using TupleTools: flatten

export PermutationMatrix
export FlowModel, CompiledFlowModel, compile, nr_params, getlayers
export AdditiveCouplingLayer, InputLayer, PermutationLayer
export PlanarFlow, RadialFlow
export Flow, FlowApproximation

include("algebra/permutation_matrix.jl")

include("models/abstract.jl")
include("models/planar_flow.jl")
include("models/radial_flow.jl")
include("models/additive_coupling_layer.jl")
include("models/input_layer.jl")
include("models/permutation_layer.jl")
include("models/flow_model.jl")

include("node.jl")
include("rules.jl")

end
