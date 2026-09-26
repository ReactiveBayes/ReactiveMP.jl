"""
    DeltaFn{F}

The Delta node, `out = f(in₁, …, inₙ)` for a deterministic function `f` of type `F`. It is a
deterministic node: its density is the Dirac delta `δ(out - f(in₁, …, inₙ))`.

Its interfaces are `out` and the group `in`, one member per input. Inputs connected to constants
or data are folded into the function (`static_inputs = :fold`), so the group `in` holds the
remaining, random, inputs only, and `(:in, k)` counts those. As for every deterministic node, its
clusters are `out` and the joint over its inputs, `(:in,)`, and it takes no factorisation.

The type carries no function: the engine is given `f` when it creates the node (`factornode(…;
nodefn = f)`) and hands it to a rule as [`getnodefn`](@extref MessagePassingRulesBase.getnodefn)`(ctx.node,
Target(:out))`, with the static inputs already folded in.

Its rules run only under a [`DeltaApproximation`](@ref), which a model must give: under the
default, [`DefaultAlgorithm`](@extref MessagePassingRulesBase.DefaultAlgorithm), no rule is found.
There is no average energy: the node's term in the Bethe free energy is minus the entropy of the
joint over its inputs, which the engine computes from that marginal.

See also [`DeltaApproximation`](@ref), [`CVIProjection`](@ref).
"""
struct DeltaFn{F} end

@define_factor_node(node = DeltaFn, type = Deterministic, interfaces = [:out, :in...], static_inputs = :fold)

"""
    is_delta_node_compatible(method) -> Val{Bool}

Whether [`DeltaApproximation`](@ref) accepts the approximation method `method`: `Val(true)` for
[`Unscented`](@extref MessagePassingRulesApproximations.Unscented) and
[`Linearization`](@extref MessagePassingRulesApproximations.Linearization), and for
[`CVIProjection`](@ref) once ExponentialFamilyProjection is loaded; `Val(false)` for anything
else. Every constructor of `DeltaApproximation` checks it.

A package that provides a new method, with its rules for [`DeltaFn`](@ref), opts the method in by
adding a method of this function that returns `Val(true)`.
"""
is_delta_node_compatible(method) = Val(false)
is_delta_node_compatible(::Unscented) = Val(true)
is_delta_node_compatible(::Linearization) = Val(true)

"""
    DeltaApproximation(; method, inverse = nothing)
    DeltaApproximation(method, inverse)

The algorithm of [`DeltaFn`](@ref): the approximation method its rules push the messages through
the function with, and an optional known inverse of the function. A model must give it; the node
has no rules without it.

# Keywords

- `method`: the approximation method, required, with no default. One of
  - [`Unscented`](@extref MessagePassingRulesApproximations.Unscented)`()`, the unscented
    transform, for normal messages;
  - [`Linearization`](@extref MessagePassingRulesApproximations.Linearization)`()`, the
    first-order expansion at the inputs' means, for normal messages;
  - [`CVIProjection`](@ref)`()`, sampling and projection onto an exponential family, for any
    messages; its rules are in a package extension, loaded with `using
    ExponentialFamilyProjection`.
- `inverse`: the known inverse of the function towards the inputs, or `nothing`. Default
  `nothing`. A single function serves every input; a tuple gives one per input, in order.
  The inverse towards input `k` takes the value of `out` followed by the other inputs, in order:
  `inverse[k](out, in₁, …, inₖ₋₁, inₖ₊₁, …, inₙ)`.

The inverse decides the node's dependencies, and so the messages towards an input:

- without an inverse, the message towards `(:in, k)` is the input's share of the joint over the
  inputs, `q[(:in,)]`, divided by that input's own message, `m[:in][k]`. The joint is the forward
  statistics through the function smoothed with the message from `out`;
- with an inverse, the message towards `(:in, k)` is the message from `out` and the other inputs'
  messages pushed through the inverse, `m[:out]` and `m[:in][!k]`, by the same method.

The message towards `out` pushes the inputs' messages through the function in both cases.
[`CVIProjection`](@ref) does not use an inverse, and ignores one with a warning.

# Throws

An `ArgumentError` when `method` is not one the node accepts
([`is_delta_node_compatible`](@ref)); for `CVIProjection` without ExponentialFamilyProjection
loaded, the error says to load it.

# Examples

```jldoctest
julia> DeltaApproximation(method = Linearization()) isa DeltaApproximation
true

julia> DeltaApproximation(method = Unscented(), inverse = (y -> y - 1,)) isa DeltaApproximation
true

julia> DeltaApproximation(method = 1)
ERROR: ArgumentError: `1` is not an approximation method of the Delta node. It takes `Unscented()` and `Linearization()` from MessagePassingRulesApproximations, and `CVIProjection()` once ExponentialFamilyProjection is loaded.
```

See also [`DeltaFn`](@ref), [`CVIProjection`](@ref).
"""
struct DeltaApproximation{M, I} <: AbstractAlgorithm
    method::M
    inverse::I

    function DeltaApproximation(method::M, inverse::I) where {M, I}
        is_delta_node_compatible(method) === Val(true) || throw(ArgumentError(incompatible_method_message(method)))
        check_inverse(method, inverse)
        return new{M, I}(method, inverse)
    end
end

DeltaApproximation(; method, inverse = nothing) = DeltaApproximation(method, inverse)

"""
    delta_method_hint(method) -> Union{Nothing, String}

A sentence on what to do about a `method` the Delta node does not accept, which the error of
[`DeltaApproximation`](@ref) appends, or `nothing`. A method whose rules live in a package
extension, such as [`CVIProjection`](@ref), names the package to load.
"""
delta_method_hint(method) = nothing

# A method that has no use for a known inverse says so when given one.
check_inverse(method, inverse) = nothing

function incompatible_method_message(method)
    message = "`$method` is not an approximation method of the Delta node. It takes `Unscented()` and " *
        "`Linearization()` from MessagePassingRulesApproximations, and `CVIProjection()` once " *
        "ExponentialFamilyProjection is loaded."
    hint = delta_method_hint(method)
    return hint === nothing ? message : string(message, " ", hint)
end

"""
    UnknownInverse

The type of a [`DeltaApproximation`](@ref) without a known inverse, `DeltaApproximation{<:Any,
Nothing}`. Its dependencies read the joint over the inputs towards an input:
`(:in, k) => (m[:in][k], q[(:in,)])`.
"""
const UnknownInverse = DeltaApproximation{<:Any, Nothing}

"""
    KnownInverse

The type of a [`DeltaApproximation`](@ref) with a known inverse, a function or a tuple of
functions, one per input. Its dependencies read the message from `out` and the other inputs'
messages towards an input: `(:in, k) => (m[:out], m[:in][!k])`.
"""
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
