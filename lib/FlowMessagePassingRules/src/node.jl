@doc raw"""
    Flow

The deterministic node ``y = f(x)``, for an invertible flow model ``f``, a
[`CompiledFlowModel`](@ref) of dimension ``d``:

```math
p(y \mid x) = \delta\big(y - f(x)\big), \qquad x, y \in \mathbb{R}^d.
```

# Interfaces

- `out`: the output ``y``, multivariate;
- `in`: the input ``x``, multivariate, of the model's dimension.

# Algorithm

The rules run under [`FlowApproximation`](@ref), which carries the model and the method, and
which the model must give. The node declares no algorithm of its own, so under
[`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm) no rule exists and a call
throws a `RuleNotFoundError`.

# Rules

Belief propagation only, one message rule per edge and per method, each taking a multivariate
normal message on the other edge and returning a multivariate normal:

- towards `out`, the message on `in` pushed forwards through ``f``;
- towards `in`, the message on `out` pulled back through ``f^{-1}``.

Under [`Linearization`](@extref MessagePassingRulesApproximations.Linearization) the model's own
Jacobian ``J`` gives ``\mathcal{N}(f(\mu), J \Sigma J^\top)``; the rules take
`MvNormalMeanCovariance`, `MvNormalMeanPrecision` and `MvNormalWeightedMeanPrecision`, and
return an `MvNormalMeanCovariance` for the first and an `MvNormalMeanPrecision` for the others. Under
[`Unscented`](@extref MessagePassingRulesApproximations.Unscented) the sigma points give the
moments; the rules take any multivariate normal and return an `MvNormalMeanCovariance`.

The node has no marginal rule, since the marginal of `in` is the product of its two messages, and
no average energy, since a deterministic node needs none: its contribution to the Bethe free
energy is minus the entropy of `q(in)`, which is exact only as far as the approximation is.

# Examples

```jldoctest; setup = :(using FlowMessagePassingRules, MessagePassingRulesBase, BayesBase, ExponentialFamily)
julia> model = compile(FlowModel(2, (AdditiveCouplingLayer(PlanarFlow(); permute = false),)), [1.0, 2.0, 3.0]);

julia> y = [1.0, 3.0 + tanh(5.0)];   # f([1.0, 2.0])

julia> result = @call_message_update_rule(
           node = Flow, target = :in, algorithm = FlowApproximation(model),
           m = (out = MvNormalMeanPrecision(y, [1.0 0.0; 0.0 1.0]),),
       );

julia> mean(getresult(result)) ≈ [1.0, 2.0]
true
```

See also [`FlowModel`](@ref), [`compile`](@ref).
"""
struct Flow end

"""
    FlowApproximation(model::CompiledFlowModel; method = Linearization())
    FlowApproximation(model::CompiledFlowModel, method)

[`Flow`](@ref)'s algorithm: the compiled flow model, and the method that pushes a normal through
it. A model using the node must give it; the node has no default algorithm.

# Arguments

- `model`: the compiled flow, a [`CompiledFlowModel`](@ref) from [`compile`](@ref).

# Keywords

- `method`: [`Linearization`](@extref MessagePassingRulesApproximations.Linearization)`()`,
  which uses the model's own Jacobians, or
  [`Unscented`](@extref MessagePassingRulesApproximations.Unscented), which uses sigma points.
  `Unscented()` takes its dimension from each input; `Unscented(dim)` fixes it, and a rule throws
  a `DimensionMismatch` when `dim` is not the model's dimension. Default `Linearization()`.

# Examples

```jldoctest; setup = :(using FlowMessagePassingRules, MessagePassingRulesApproximations)
julia> model = compile(FlowModel(2, (AdditiveCouplingLayer(PlanarFlow()),)), [1.0, 2.0, 3.0]);

julia> FlowApproximation(model).method isa Linearization
true

julia> FlowApproximation(model; method = Unscented(2)).method isa Unscented
true
```
"""
struct FlowApproximation{M <: AbstractCompiledFlowModel, A <: Union{Linearization, Unscented}} <: AbstractAlgorithm
    model::M
    method::A
end

FlowApproximation(model::AbstractCompiledFlowModel; method::Union{Linearization, Unscented} = Linearization()) = FlowApproximation(model, method)

"""
    getmodel(algo::FlowApproximation) -> CompiledFlowModel

The compiled flow model of a [`FlowApproximation`](@ref).
"""
getmodel(algo::FlowApproximation) = algo.model

"""
    getmethod(algo::FlowApproximation)

The method of a [`FlowApproximation`](@ref), a `Linearization` or an `Unscented`.
"""
getmethod(algo::FlowApproximation) = algo.method

@define_factor_node(node = Flow, type = Deterministic, interfaces = [:out, :in])
