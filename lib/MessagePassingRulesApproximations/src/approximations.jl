"""
    AbstractApproximationMethod

Supertype of the moment-propagation methods, such as [`Unscented`](@ref).
"""
abstract type AbstractApproximationMethod end

"""
    approximation_name(method)

The method's full name, for display.
"""
function approximation_name end

"""
    approximation_short_name(method)

The method's abbreviated name, for display.
"""
function approximation_short_name end
