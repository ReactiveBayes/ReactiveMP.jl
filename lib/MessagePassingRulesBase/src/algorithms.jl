"""
    AbstractAlgorithm

Supertype of every algorithm. An algorithm selects which rules run and carries their
parameters; a rule reads its value through the `algo` body slot.
"""
abstract type AbstractAlgorithm end

"""
    BP()

Belief propagation.
"""
struct BP <: AbstractAlgorithm end

"""
    VMP()

Variational message passing.
"""
struct VMP <: AbstractAlgorithm end

"""
    ispure(algorithm::Type{<:AbstractAlgorithm})

Whether rules under this algorithm are pure unless they say otherwise.

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
