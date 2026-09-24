"""
    Flow

The deterministic node `out = f(in)`, for an invertible flow model `f`, a
[`CompiledFlowModel`](@ref). Its interfaces are `out` and `in`, both multivariate, and its rules
run under [`FlowApproximation`](@ref), which carries the model and which the model must give:
the node declares no algorithm of its own, so under `DefaultAlgorithm` no rule exists.
"""
struct Flow end

"""
    FlowApproximation(model; method = Linearization())

[`Flow`](@ref)'s algorithm: the compiled flow model and how a normal is pushed through it,
`Linearization()` by the model's own Jacobians, or `Unscented()` by sigma points. `Unscented(dim)`
fixes the dimension; `Unscented()` takes it from the input. v6 called it `FlowMeta(model,
approximation)`.
"""
struct FlowApproximation{M <: AbstractCompiledFlowModel, A <: Union{Linearization, Unscented}} <: AbstractAlgorithm
    model::M
    method::A
end

FlowApproximation(model::AbstractCompiledFlowModel; method::Union{Linearization, Unscented} = Linearization()) = FlowApproximation(model, method)

getmodel(algo::FlowApproximation) = algo.model
getmethod(algo::FlowApproximation) = algo.method

@define_factor_node(node = Flow, type = Deterministic, interfaces = [:out, :in])
