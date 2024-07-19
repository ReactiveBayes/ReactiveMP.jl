export Linearization

"""
The `Linearization` structure defines the approximation method of the `Delta` and `Flow` factor nodes. 
This method performs a local linearization of f around expansion point x.

The `Linearization` structure with default parameters can be constructed as `Linearization()`.

The `Linearization` structure is used inside the `DeltaMeta` or `FlowMeta` structures and can be included as:
```
    y ~ f(x) where { meta = DeltaMeta(method = Linearization()) }
    # or
    y ~ Flow(x) where { meta = FlowMeta(flowmodel, Linearization()) }
```
"""
struct Linearization <: AbstractApproximationMethod end

is_delta_node_compatible(::Linearization) = Val(true)

# ported from ForneyLab.jl
using ForwardDiff

function approximate(::Linearization, g::G, x_hat) where {G}
    return local_linearization(g, x_hat)
end

"""
    local_linearization(g, x)

Returns linear components `(a, b)` for the function `g` at the point `x`.
"""
function local_linearization end

local_linearization(g::G, x_hat::Tuple{T}) where {G, T} = local_linearization(g(first(x_hat)), g, x_hat)

"""Return local linearization of g around expansion point x_hat for Delta node with single input interface and univariate output"""
function local_linearization(result::R, g::G, x_hat::Tuple{AbstractVector{T}}) where {R <: Real, G, T <: Real}
    a = ForwardDiff.gradient(g, first(x_hat))'
    b = result - a * first(x_hat)
    return (a, b)
end

"""Return local linearization of g around expansion point x_hat for Delta node with single input interface and univariate output"""
function local_linearization(result::R, g::G, x_hat::Tuple{T}) where {R <: Real, G, T}
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
    lx_ds = size.(x_hat)

    splitg = let g = g, ds = lx_ds
        (x) -> g(__splitjoin(x, ds)...)
    end

    return local_linearization(g(x_hat...), splitg, g, x_hat)
end

# In case if `g(x_hat)` returns a number and inputs are numbers too
function local_linearization(r::Real, splitg::S, g::G, x_hat) where {S, G}
    return local_linearization(r, splitg, g, (g, lx_hat) -> (ForwardDiff.gradient(splitg, lx_hat)::Vector{eltype(lx_hat)})', x_hat)
end

# In case if `g(x_hat)` returns a vector, but inputs are numbers
function local_linearization(r::AbstractVector, splitg::S, g::G, x_hat) where {S, G}
    return local_linearization(r, splitg, g, (g, lx_hat) -> (ForwardDiff.jacobian(splitg, lx_hat)::Matrix{eltype(lx_hat)}), x_hat)
end

function local_linearization(r, splitg::S, g::G, fA::F, x_hat) where {S, G, F}
    lx_hat = __as_vec(x_hat)
    # `fA` calls either `gradient` or `jacobian`, depending on the type of the `r`
    A = fA(splitg, lx_hat)
    b = r - A * lx_hat
    return (A, b)
end

# Approximation methods extensions for Normal distributions family

# This function extends the `Linearization` approximation method in case if all inputs are from the `NormalDistributionsFamily`
function approximate(method::Linearization, f::F, distributions::NTuple{N, NormalDistributionsFamily}) where {F, N}

    # Collect statistics for the inputs of the function `f`
    statistics = mean_cov.(distributions)
    means      = first.(statistics)
    covs       = last.(statistics)

    # Compute the local approximation for the function `f`
    (A, b) = approximate(method, f, means)

    # Execute the 'joint' message in the linearized version of `f`
    joint       = convert(JointNormal, means, covs)
    jmean, jcov = mean_cov(joint)

    m = A * jmean + b
    V = A * jcov * A'

    return convert(promote_variate_type(typeof(m), NormalMeanVariance), m, V)
end
