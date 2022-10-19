export Linearization

struct Linearization <: AbstractApproximationMethod end

# ported from ForneyLab.jl

using LazyArrays
using ForwardDiff

as_vec(d::Float64) = [d] # Extend vectorization to Float
as_vec(something)  = vec(something) # Avoid type-piracy, but better to refactor this

as_mat(d::Float64) = [d;;]
as_mat(mat::AbstractMatrix) = mat

# This function computes cumulative start indices in the tuple `sizes`, note that `()` acts as size 1`
function __linearization_starts_at(sizes::Tuple)
    return last(reduce(sizes; init = (0, ())) do result, size
        current, accumulated = result
        next = current + prod(size)
        return (next, (accumulated..., current + 1))
    end)
end

@propagate_inbounds __as_vec_copyto!(container, start, input::Real)             = container[start] = input
@propagate_inbounds __as_vec_copyto!(container, start, input::AbstractVecOrMat) = copyto!(container, start, input, 1, length(input))

# This function linearizes the `inputs` argument into one (potentially big) vector
# For example (1, 2) becomes `[ 1, 2 ]`
#             ([2, 3], 2) becomes `[ 2, 3, 2 ]`
#             ([2, 3], [ 1.0 0.0; 0.0 1.0 ]) becomes `[ 2.0, 3.0, 1.0, 0.0, 0.0, 1.0 ]`
#             and so on
function __linearization_as_vec(inputs::Tuple)
    sizes     = prod.(size.(inputs))
    starts_at = __linearization_starts_at(sizes)
    total     = last(starts_at) + prod(last(sizes)) - 1

    T = promote_type(eltype.(inputs)...)
    x = Vector{T}(undef, total)

    foreach(inputs, starts_at) do input, start
        @inbounds __as_vec_copyto!(x, start, input)
    end

    return x
end

# This function extracts elements from the linearized version of inputs from the `__linearization_as_vec` function
# In case of `size = ()` we extract a single element a treat it as it would be a Univariate variable
function __linearization_splitjoinelement(x::AbstractVector, start, size::Tuple{})
    return x[start]
end

# This function extracts elements from the linearized version of inputs from the `__linearization_as_vec` function
# In case of `size = (Int, )` we extract the specified amount of elements as a view and treat it as a Multivariate variable
function __linearization_splitjoinelement(x::AbstractVector, start, size::Tuple{Int})
    return view(x, start:(start + prod(size) - 1))
end

# This function extracts elements from the linearized version of inputs from the `__linearization_as_vec` function
# In case of `size = (Int, Int)` we extract the specified amount of elements, reshape it into a matrix and treat it as a Matrixvariate variable
function __linearization_splitjoinelement(x::AbstractVector, start, size::Tuple{Int, Int})
    return reshape(view(x, start:(start + prod(size) - 1)), first(size), last(size))
end

# This function extracts elements from the linearized version of inputs from the `__linearization_as_vec` function all at once
# Essentially this is the `undo` operation for the `__linearization_as_vec`
# See also `__linearization_splitjoinelement`
function __linearization_splitjoin(x::AbstractVector, sizes::Tuple)
    return map(sizes, __linearization_starts_at(sizes)) do size, start 
        return __linearization_splitjoinelement(x, start, size)
    end
end

"""
    local_linearization(g, x)

Returns linear components `(a, b)` for the function `g` at the point `x`.
"""
function local_linearization end

local_linearization(g::G, x_hat::Tuple{T}) where { G, T } = local_linearization(g(first(x_hat)), g, x_hat)

"""Return local linearization of g around expansion point x_hat for Delta node with single input interface and univariate output"""
function local_linearization(result::R, g::G, x_hat::Tuple{T}) where { R <: Real, G, T }
    a = ForwardDiff.derivative(g, first(x_hat))
    b = result - a * first(x_hat)
    return (a, b)
end

"""Return local linearization of g around expansion point x_hat for Delta node with single input interface and multivariate output"""
function local_linearization(result::AbstractVector, g::G, x_hat::Tuple{T}) where {G, T}
    A = ForwardDiff.jacobian(g, first(x_hat))
    b = result - A * first(x_hat)
    return (A, b)
end

"""Return local linearization of g around expansion point x_hat for Delta node with multiple input interfaces."""
function local_linearization(g::G, x_hat::Tuple) where {G}
    lx_ds  = size.(x_hat)

    splitg = let g = g, ds = lx_ds
        (x) -> g(__linearization_splitjoin(x, ds)...)
    end

    return localLinearizationMultiIn(g(x_hat...), splitg, g, x_hat)
end

# In case if `g(x_hat)` returns a number and inputs are numbers too
function localLinearizationMultiIn(r::Real, splitg::S, g::G, x_hat) where { S, G }
    return localLinearizationMultiIn(r, splitg, g, (g, lx_hat) -> (ForwardDiff.gradient(splitg, lx_hat)::Vector{eltype(lx_hat)})', x_hat)
end

# In case if `g(x_hat)` returns a vector, but inputs are numbers
function localLinearizationMultiIn(r::AbstractVector, splitg::S, g::G, x_hat) where { S, G }
    return localLinearizationMultiIn(r, splitg, g, (g, lx_hat) -> (ForwardDiff.jacobian(splitg, lx_hat)::Matrix{eltype(lx_hat)}), x_hat)
end

function localLinearizationMultiIn(r, splitg::S, g::G, fA::F, x_hat) where {S, G, F}
    lx_hat = __linearization_as_vec(x_hat)
    # `fA` calls either `gradient` or `jacobian`, depending on the type of the `r`
    A = fA(splitg, lx_hat)
    b = r - A * lx_hat
    return (A, b)
end