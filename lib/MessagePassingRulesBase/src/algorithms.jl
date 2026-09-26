"""
    AbstractAlgorithm

Supertype of every algorithm. An algorithm selects which rules run and carries their
parameters, which a rule reads through its `algo` slot. It is **not** an inference scheme:
belief propagation, variational message passing and their structured forms all come from the
factorisation, under one [`DefaultAlgorithm`](@ref).

A custom algorithm is either a rule switcher, when someone wants a different set of rules, or
a node's own algorithm, when the node needs one. It comes in two kinds:
- a direct subtype of `AbstractAlgorithm` **stands alone**: only its own rules and
  dependencies apply to it;
- a subtype of [`DefaultAlgorithmExtension`](@ref) **extends the default**: where it defines
  no rule or dependencies of its own, those of `DefaultAlgorithm` apply.

An algorithm with parameters stores them as fields, and a rule reads them from its `algo` slot.

```julia
struct MyVMP{T} <: AbstractAlgorithm
    iterations::Int
    tolerance::T
end

@define_message_update_rule(
    node = MyNode, target = :out, algorithm = MyVMP,
    args = (q[:in]::Any,),
    body = (algo, args) -> solve(args.q[:in]; maxiter = algo.iterations, tol = algo.tolerance),
)
```

See also [`ispure`](@ref), [`@define_dependencies`](@ref).
"""
abstract type AbstractAlgorithm end

"""
    DefaultAlgorithm()

The algorithm every node runs under unless it declares its own: Bethe free energy
minimisation. Whether a rule behaves as belief propagation, variational message passing or
their structured form depends on the factorisation, through the engine's default dependency
scheme, not on the algorithm. A rule that omits `algorithm` belongs to its node's default,
[`default_algorithm`](@ref)`(node)`, which is this unless the node declares otherwise.
"""
struct DefaultAlgorithm <: AbstractAlgorithm end

"""
    DefaultAlgorithmExtension

Supertype of the algorithms that extend [`DefaultAlgorithm`](@ref). Resolution looks for the
extension's own rule first and falls back to the default's, and likewise for dependencies.
A rule reached through that fallback receives `DefaultAlgorithm()` in its `algo` slot, the
algorithm it was written for. The fallback is a second lookup, not dispatch on a supertype:
if default rules dispatched on an abstract type, an extension's rule with broader inputs than
a default one would be ambiguous with it, and resolution must never throw.

```julia
struct MyRules <: DefaultAlgorithmExtension end   # override some rules, inherit the rest
```
"""
abstract type DefaultAlgorithmExtension <: AbstractAlgorithm end

# Whether a rule defined under `rule_algorithm` can serve a call made under `algorithm`,
# directly or, for an extension, through the fallback to the default.
admits(algorithm, rule_algorithm::Type) =
    algorithm isa rule_algorithm || (algorithm isa DefaultAlgorithmExtension && rule_algorithm === DefaultAlgorithm)

"""
    ispure(algorithm::Type{<:AbstractAlgorithm}) -> Bool
    ispure(algorithm::AbstractAlgorithm) -> Bool

Whether rules under this algorithm are pure unless they say otherwise. An impure algorithm
adds a method, `MessagePassingRulesBase.ispure(::Type{MyAlgorithm}) = false`.

A pure rule mutates neither its inputs nor any state shared beyond one call, such as fields
of its algorithm. It may write to its own output buffer and to scratch storage it owns, so
in-place rules can be pure. Randomness comes from `ctx.rng`, which the caller owns; an
algorithm that carries its own random number generator, or any other state that persists
between calls, is impure.

Purity is declared, not proved: the default is `true`. A rule may override its algorithm
with `pure = false`, and an engine auditing purity must read the rule's own flag, so the
override cannot hide impurity. Purity is an audit policy, for threading and for finding side
effects; it says nothing about whether a rule can be differentiated, which is tested
separately.
"""
ispure(::Type{<:AbstractAlgorithm}) = true
ispure(algorithm::AbstractAlgorithm) = ispure(typeof(algorithm))
