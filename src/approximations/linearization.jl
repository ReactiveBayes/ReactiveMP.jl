export Linearization

struct Linearization <: AbstractApproximationMethod end

import Base: split, vec

# ported from ForneyLab.jl

using ForwardDiff

as_vec(d::Float64) = [d] # Extend vectorization to Float
as_vec(something)  = vec(something) # Avoid type-piracy, but better to refactor this

"""
Return integer dimensionality
"""
intdim(tup::Tuple) = prod(tup) # Returns 1 for ()

"""
Split a vector in chunks of lengths specified by ds.
"""
function linearizationSplit(vec::Vector, ds::Vector{<:Tuple})
    N = length(ds)
    res = Vector{Any}(undef, N)

    d_start = 1
    for k in 1:N # For each original statistic
        d_end = d_start + intdim(ds[k]) - 1

        if ds[k] == () # Univariate
            res[k] = vec[d_start] # Return scalar
        else # Multi- of matrix variate
            res[k] = reshape(vec[d_start:d_end], ds[k]) # Return vector or matrix
        end

        d_start = d_end + 1
    end

    return res
end

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
function localLinearizationSingleIn(g, x_hat::AbstractVector{T}) where { T <: Real }
    A = ForwardDiff.jacobian(g, x_hat)
    b = g(x_hat) - A * x_hat
    return (A, b)
end

"""
Return local linearization of g around expansion point x_hat
for Delta node with multiple input interfaces
"""
function localLinearizationMultiIn(g, x_hat::AbstractVector{T}) where { T <: Real }
    g_unpacked(x) = g(x...)
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
    g_unpacked(x) = g(linearizationSplit(x, ds)...)
    A = ForwardDiff.jacobian(g_unpacked, x_cat)
    b = g(x_hat...) - A * x_cat
    return (A, b)
end
