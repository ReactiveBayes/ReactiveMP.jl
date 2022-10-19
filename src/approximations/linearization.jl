export Linearization

struct Linearization <: AbstractApproximationMethod end

# ported from ForneyLab.jl

using ForwardDiff

as_vec(d::Float64) = [d] # Extend vectorization to Float
as_vec(something)  = vec(something) # Avoid type-piracy, but better to refactor this

"""
Concatenate a vector (of vectors and floats) and return with original dimensions (for splitting)
"""
function linearizationConcatenate(xs::AbstractVector)
    ds = size.(xs) # Extract dimensions
    x = vcat(as_vec.(xs)...)
    return (x, ds)
end

"""
Return local linearization of g around expansion point x_hat
for Delta node with single (univariate) input interface
"""
function localLinearizationSingleIn(g, x_hat::Real)
    a = ForwardDiff.derivative(g, x_hat)
    b = g(x_hat) - a * x_hat
    return (a, b)
end

"""
Return local linearization of g around expansion point x_hat
for Delta node with single (multivariate) input interface
"""
function localLinearizationSingleIn(g, x_hat::AbstractVector{T}) where {T <: Real}
    A = ForwardDiff.jacobian(g, x_hat)
    b = g(x_hat) - A * x_hat
    return (A, b)
end

"""
Return local linearization of g around expansion point x_hat
for Delta node with multiple input interfaces
"""
function localLinearizationMultiIn(g, x_hat::AbstractVector{T}) where {T <: Real}
    g_unpacked = let g = g
        (x) -> g(x...)
    end

    A = ForwardDiff.gradient(g_unpacked, x_hat)'
    b = g(x_hat...) - A * x_hat
    return (A, b)
end

"""
Return local linearization of g around expansion point x_hat
for Delta node with multiple input interfaces
"""
function localLinearizationMultiIn(g::Any, x_hat::AbstractVector)
    (x_cat, ds) = linearizationConcatenate(x_hat)

    g_unpacked = let ds = ds, g = g
        (x) -> g(splitjoint(x, ds)...)
    end

    A = ForwardDiff.jacobian(g_unpacked, x_cat)
    b = g(x_hat...) - A * x_cat
    return (A, b)
end
