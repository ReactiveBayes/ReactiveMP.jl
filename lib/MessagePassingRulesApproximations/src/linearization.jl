"""
    Linearization()

The local-linearization method: a function is replaced by its first-order Taylor expansion at
the inputs' means, computed by automatic differentiation (ForwardDiff), and moments pass
through the linear map exactly. It has no parameters.

It is exact for affine functions and accurate when the function is close to linear over the
inputs' spread; the function must be differentiable, and written generically enough for
ForwardDiff's dual numbers. [`Unscented`](@ref) is the alternative that needs no derivatives.

See [`approximate`](@ref)`(::Linearization, g, x̂)` for the map it computes.
"""
struct Linearization <: AbstractApproximationMethod end

approximation_name(::Linearization) = "Linearization"
approximation_short_name(::Linearization) = "LN"

"""
    approximate(::Linearization, g, x̂::Tuple) -> (A, b)

The local linear map of `g` at the expansion point `x̂`, so that `g(x) ≈ A * x + b` near `x̂`.

# Arguments

- `g`: a function of one or several arguments, each a number or a vector, returning a number or
  a vector;
- `x̂`: the expansion point, a tuple with one entry per argument of `g`, typically the inputs'
  means.

# Returns

`(A, b)`, where `x` stands for the arguments concatenated into one vector. `A` is a derivative
for a scalar function of one scalar, a gradient row for a scalar function of a vector, and a
Jacobian otherwise; `b = g(x̂) - A * x̂`. Propagating a normal `N(m, V)` of `x` gives
`N(A * m + b, A * V * A')`.

# Examples

```jldoctest; setup = :(using MessagePassingRulesApproximations)
julia> approximate(Linearization(), (x, y) -> x .- y, ([1.0, 2.0], [0.5, 0.5]))
([1.0 0.0 -1.0 0.0; 0.0 1.0 0.0 -1.0], [0.0, 0.0])
```
"""
approximate(::Linearization, g::G, x̂::Tuple) where {G} = local_linearization(g, x̂)

"""
    local_linearization(g, x̂::Tuple) -> (A, b)

The first-order expansion of `g` at `x̂`, the function behind
[`approximate`](@ref)`(::Linearization, g, x̂)`, which documents its arguments and result.
"""
function local_linearization end

local_linearization(g::G, x̂::Tuple{T}) where {G, T} = local_linearization(g(first(x̂)), g, x̂)

# One input. A scalar function of a vector has a gradient, of a scalar a derivative; a vector
# function has a Jacobian, or a derivative when its input is a scalar.
function local_linearization(result::Real, g::G, x̂::Tuple{AbstractVector{<:Real}}) where {G}
    A = ForwardDiff.gradient(g, first(x̂))'
    return (A, result - A * first(x̂))
end

function local_linearization(result::Real, g::G, x̂::Tuple{T}) where {G, T}
    A = ForwardDiff.derivative(g, first(x̂))
    return (A, result - A * first(x̂))
end

function local_linearization(result::AbstractVector, g::G, x̂::Tuple{T}) where {G, T}
    A = ForwardDiff.jacobian(g, first(x̂))
    return (A, result - A * first(x̂))
end

function local_linearization(result::AbstractVector, g::G, x̂::Tuple{<:Real}) where {G}
    A = ForwardDiff.derivative(g, first(x̂))
    return (A, result - A * first(x̂))
end

# Several inputs: `g` of their concatenation, split back into its arguments.
function local_linearization(g::G, x̂::Tuple) where {G}
    sizes = size.(x̂)
    splitg = let g = g, sizes = sizes
        x -> g(__splitjoin(x, sizes)...)
    end
    return local_linearization_split(g(x̂...), splitg, x̂)
end

function local_linearization_split(result::Real, splitg::S, x̂::Tuple) where {S}
    x = __as_vec(x̂)
    A = (ForwardDiff.gradient(splitg, x)::Vector{eltype(x)})'
    return (A, result - A * x)
end

function local_linearization_split(result::AbstractVector, splitg::S, x̂::Tuple) where {S}
    x = __as_vec(x̂)
    A = ForwardDiff.jacobian(splitg, x)::Matrix{eltype(x)}
    return (A, result - A * x)
end
