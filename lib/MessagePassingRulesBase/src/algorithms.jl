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

Whether rules under this algorithm are pure unless they say otherwise. A pure rule mutates
neither its inputs nor shared algorithm state; writes to its own output and scratch storage
are allowed. Purity is declared, not proved: the default is `true`, and an algorithm that
keeps state, including its own random number generator, must return `false`.
"""
ispure(::Type{<:AbstractAlgorithm}) = true
ispure(algorithm::AbstractAlgorithm) = ispure(typeof(algorithm))
