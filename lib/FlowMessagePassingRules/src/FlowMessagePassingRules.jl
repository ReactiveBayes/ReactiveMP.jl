"""
    FlowMessagePassingRules

The [`Flow`](@ref) node, the deterministic node ``y = f(x)`` for an invertible flow model `f`,
and the flow models it is built from. A model is a tuple of layers, [`FlowModel`](@ref), given
its parameters by [`compile`](@ref); the layers are [`InputLayer`](@ref),
[`AdditiveCouplingLayer`](@ref) and [`PermutationLayer`](@ref), and a coupling layer holds a
coupling flow, [`PlanarFlow`](@ref) or [`RadialFlow`](@ref). [`PermutationMatrix`](@ref) is the
permutation a [`PermutationLayer`](@ref) applies.

The node's rules push a multivariate normal through the compiled model, forwards towards `out`
and backwards, through the inverse, towards `in`, by linearisation or by the unscented
transform. They run under [`FlowApproximation`](@ref), which carries the model, and which the
model must give: under [`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm)
no rule exists.

# Examples

```jldoctest; setup = :(using FlowMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily)
julia> model = compile(FlowModel(2, (AdditiveCouplingLayer(PlanarFlow(); permute = false),)), [1.0, 2.0, 3.0]);

julia> result = @call_message_update_rule(
           node = Flow, target = :out, algorithm = FlowApproximation(model),
           m = (in = MvNormalMeanCovariance([1.0, 2.0], [1.0 0.0; 0.0 1.0]),),
       );

julia> μ, Σ = mean_cov(getresult(result));

julia> μ ≈ [1.0, 3.0 + tanh(5.0)]   # y₂ = x₂ + x₁ + tanh(2x₁ + 3) at the mean
true

julia> J = [1.0 0.0; 1 + 2 * (1 - tanh(5.0)^2) 1.0];   # the Jacobian at the mean

julia> Σ ≈ J * J'
true
```
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
