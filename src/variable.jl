"""
    AbstractVariable

An abstract supertype for all variable types in the factor graph.
Concrete subtypes include:
- [`ReactiveMP.RandomVariable`](@ref)
- [`ReactiveMP.ConstVariable`](@ref)
- [`ReactiveMP.DataVariable`](@ref).
"""
abstract type AbstractVariable end
