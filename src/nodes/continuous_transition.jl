export transfominator, CTransition, ContinuousTransition, CTMeta

import LazyArrays
import StatsFuns: log2π

@doc raw"""
The ContinuousTransition node transforms an m-dimensional (dx) vector x into an n-dimensional (dy) vector y via a linear (or nonlinear) transformation with a `n×m`-dimensional matrix `A` that is constructed from a `nm`-dimensional vector `a`.

To construct the matrix A, the elements of `a` are filled into A according to the transformation function provided with meta.

Check CTMeta for more details on how to specify the transformation function that **must** return a matrix.

```julia
y ~ ContinuousTransition(x, a, W)
```
Interfaces:
1. y - n-dimensional output of the ContinuousTransition node.
2. x - m-dimensional input of the ContinuousTransition node.
3. a - `nm`-dimensional vector that casts into the matrix `A`.
4. W - `n×n`-dimensional precision matrix used to soften the transition and perform variational message passing, as belief-propagation is not feasible for `y = Ax`.

Note that you can set W to a fixed value or put a prior on it to control the amount of jitter.
"""
struct ContinuousTransition end

const CTransition = ContinuousTransition

@node ContinuousTransition Stochastic [y, x, a, W]

@doc raw"""
`CTMeta` is used as a metadata flag in `ContinuousTransition` to define the transformation function for constructing the matrix `A` from vector `a`.

There are two scenarios for specifying the transformation:
1. **Linear Transformation**: In this case, `CTMeta` requires a transformation function and the length of vector `a`. 
2. **Nonlinear Transformation**: For nonlinear transformations, `CTMeta` expects a transformation function and a vector `â`, which acts as an expansion point for approximating the transformation linearly.

Constructors:
- `CTMeta(transformation::Function, len::Integer)`: Used for linear transformations.
- `CTMeta(transformation::Function, â::Vector{<:Real})`: Used for nonlinear transformations.

Fields:
- `ds`: A tuple indicating the dimensionality of the ContinuousTransition (dy, dx).
- `Fs`: Represents the masks, which can be either a Vector of AbstractMatrices or a Function, depending on the transformation type.
- `es`: A Vector of unit vectors used in the transformation process.

The `CTMeta` struct plays a pivotal role in defining how the vector `a` is transformed into the matrix `A`, thus influencing the behavior of the `ContinuousTransition` node.
"""
struct CTMeta
    ds::Tuple # dimensionality of ContinuousTransition (dy, dx)
    Fs::Union{Vector{<:AbstractMatrix}, <:Function} # masks
    es::Vector{<:AbstractVector} # unit vectors

    # NOTE: this meta is not user-friendly, I don't like supplying the length of a vector
    # perhaps making mutable struct with empty meta first will be better from user perspective
    # meta for linear transformation of a vector to a matrix
    function CTMeta(transformation::Function, len::Integer)
        dy, dx = size(transformation(zeros(len)))
        Fs = [ForwardDiff.jacobian(a -> transformation(a)[i, :], 1:len) for i in 1:dy]
        es = [StandardBasisVector(dy, i, 1.0) for i in 1:dy]
        return new((dy, dx), Fs, es)
    end

    # meta for nonlinear transformation of a vector to a matrix
    function CTMeta(transformation::Function, â::Vector{<:Real})
        dy, dx = size(transformation(â))
        es = [StandardBasisVector(dy, i, 1.0) for i in 1:dy]
        return new((dy, dx), transformation, es)
    end
end

getunits(meta::CTMeta)          = meta.es
getdimensionality(meta::CTMeta) = meta.ds

getmasks(ctmeta::CTMeta, a)                 = process_Fs(ctmeta.Fs, a)
process_Fs(Fs::Vector{<:AbstractMatrix}, a) = Fs

# NOTE: this doesn't seem to be the right way of working with nonlinar approximation
process_Fs(Fs::Function, a) = [ForwardDiff.jacobian(a -> Fs(a)[i, :], a) for i in 1:size(Fs(a), 1)]

@node ContinuousTransition Stochastic [y, x, a, W]

default_meta(::Type{CTMeta}) = error("ContinuousTransition node requires meta flag explicitly specified")

default_functional_dependencies_pipeline(::Type{<:ContinuousTransition}) = RequireMarginalFunctionalDependencies((3,), (nothing,))

function ctcompanion_matrix(a, meta::CTMeta)
    Fs, es = getmasks(meta, a), getunits(meta)
    dy, _ = getdimensionality(meta)
    A = sum(es[i] * a' * Fs[i]' for i in 1:dy)
    return A
end

@average_energy ContinuousTransition (q_y_x::MultivariateNormalDistributionsFamily, q_a::MultivariateNormalDistributionsFamily, q_W::Wishart, meta::CTMeta) = begin
    ma, Va   = mean_cov(q_a)
    myx, Vyx = mean_cov(q_y_x)
    mW       = mean(q_W)

    dy, dx = getdimensionality(meta)
    Fs, es = getmasks(meta, ma), getunits(meta)
    n      = div(ndims(q_y_x), 2)
    mA     = ctcompanion_matrix(ma, meta)
    mx, Vx = myx[(dy + 1):end], Vyx[(dy + 1):end, (dy + 1):end]
    my, Vy = myx[1:dy], Vyx[1:dy, 1:dy]
    Vyx    = Vyx[1:dy, (dy + 1):end]
    g₁     = my' * mW * my + tr(Vy * mW)
    g₂     = mx' * mA' * mW * my + tr(Vyx * mA' * mW)
    g₃     = g₂
    G      = sum(sum(es[i]' * mW * es[j] * Fs[i] * (ma * ma' + Va) * Fs[j]' for i in 1:length(Fs)) for j in 1:length(Fs))
    g₄     = mx' * G * mx + tr(Vx * G)
    AE     = n / 2 * log2π - 0.5 * mean(logdet, q_W) + 0.5 * (g₁ - g₂ - g₃ + g₄)

    return AE
end
