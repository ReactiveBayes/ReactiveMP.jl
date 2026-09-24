"""
    DeltaFn{F}

The Delta node for a function of type `F`: `DeltaFn{typeof(f)}` is the node `z = f(x₁, …, xₙ)`.
Its interfaces are `out` and the group `in`, one member per input the engine does not fold.
Inputs connected to constants or data are folded into the function (`static_inputs = :fold`),
so `(:in, k)` counts the remaining inputs only.

The node type carries no function: the engine is given the function when it creates the node,
and hands it to a rule through `getnodefn`. The node's rules need an algorithm, a
[`DeltaApproximation`](@ref); under its default, `DefaultAlgorithm`, no rule is found.
"""
struct DeltaFn{F} end

@define_factor_node(node = DeltaFn, type = Deterministic, interfaces = [:out, :in...], static_inputs = :fold)

"""
    is_delta_node_compatible(method)

`Val(true)` for an approximation method a [`DeltaApproximation`](@ref) accepts, `Val(false)`
otherwise. A package providing a method opts it in by adding a method.
"""
is_delta_node_compatible(method) = Val(false)
is_delta_node_compatible(::Unscented) = Val(true)
is_delta_node_compatible(::Linearization) = Val(true)

"""
    DeltaApproximation(; method, inverse = nothing)

[`DeltaFn`](@ref)'s algorithm: `method` approximates the pushforward through the function, and
`inverse`, when given, is the known inverse towards the input, or a tuple of them, one per
input. v6 called it `DeltaMeta`. Without an inverse, a message towards an input divides the
joint over the inputs by that input's own message; with one, it pushes the other messages
through the inverse.
"""
struct DeltaApproximation{M, I} <: AbstractAlgorithm
    method::M
    inverse::I
end

function DeltaApproximation(; method, inverse = nothing)
    is_delta_node_compatible(method) === Val(true) || throw(ArgumentError("the method `$method` is not compatible with the Delta node"))
    return DeltaApproximation(method, inverse)
end

"""The form of [`DeltaApproximation`](@ref) without a known inverse."""
const UnknownInverse = DeltaApproximation{<:Any, Nothing}

"""The form of [`DeltaApproximation`](@ref) with a known inverse, or one per input."""
const KnownInverse = DeltaApproximation{<:Any, <:Union{Function, Tuple{Vararg{Function}}}}

inverse_towards(algorithm::DeltaApproximation{<:Any, <:Function}, k) = algorithm.inverse
inverse_towards(algorithm::DeltaApproximation{<:Any, <:Tuple}, k) = algorithm.inverse[k]

# Free energy counts the joint over the inputs, whatever the inverse; the messages towards an
# input differ. Without an inverse, towards `in[k]` is the joint over the inputs divided by the
# input's own message; with one, the other messages pushed through the inverse.
@define_dependencies(
    node = DeltaFn, algorithm = UnknownInverse,
    dependencies = [:out => (m[:in...],), (:in, k) => (m[:in][k], q[(:in,)])],
    free_energy_partition = [(:out,), (:in,)],
)

@define_dependencies(
    node = DeltaFn, algorithm = KnownInverse,
    dependencies = [:out => (m[:in...],), (:in, k) => (m[:out], m[:in][!k])],
    free_energy_partition = [(:out,), (:in,)],
)
