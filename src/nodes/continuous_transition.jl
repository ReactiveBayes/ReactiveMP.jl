export CTransition, ContinuousTransition, CTMeta

import LazyArrays
import StatsFuns: log2π

@doc raw"""
The ContinuousTransition node transforms an m-dimensional (dx) vector x into an n-dimensional (dy) vector y via a linear (or nonlinear) transformation with a `n×m`-dimensional matrix `A` that is constructed from a vector `a`.

To construct the matrix A, the elements of `a` are filled into A according to the transformation function provided with meta. `a` must be of MultivariateNormalDistributionsFamily type. If you intend to use univariate Gaussian, use it as a vector of length `1``, e.g. `a ~ MvNormalMeanCovariance([0.0], [1.;])`.

Check CTMeta for more details on how to specify the transformation function that **must** return a matrix.

```julia
y ~ ContinuousTransition(x, a, W) where {meta = CTMeta(transformation, â)}
```
Interfaces:
1. y - n-dimensional output of the ContinuousTransition node.
2. x - m-dimensional input of the ContinuousTransition node.
3. a - any-dimensional vector that casts into the matrix `A`. 
4. W - `n×n`-dimensional precision matrix used to soften the transition and perform variational message passing.

Note that you can set W to a fixed value or put a prior on it to control the amount of jitter.
"""
struct ContinuousTransition end

const CTransition = ContinuousTransition

@node ContinuousTransition Stochastic [y, x, a, W]

@doc raw"""
`CTMeta` is used as a metadata flag in `ContinuousTransition` to define the transformation function for constructing the matrix `A` from vector `a`.

`CTMeta` requires a transformation function and the length of vector `a`, which acts as an expansion point for approximating the transformation linearly. If transformation appears to be linear, then no approximation is performed.

Constructors:
- `CTMeta(transformation::Function, â::Vector{<:Real})`: Constructs a `CTMeta` struct with the transformation function and allocated basis vectors.

Fields:
- `ds`: A tuple indicating the dimensionality of the ContinuousTransition (dy, dx).
- `f`:  Represents the transformation function that transforms vector `a` into matrix `A`
- `es`: A Vector of unit vectors used in the transformation process.

The `CTMeta` struct plays a pivotal role in defining how the vector `a` is transformed into the matrix `A`, thus influencing the behavior of the `ContinuousTransition` node.
"""
struct CTMeta{T <: Tuple, F <: Function, V <: Vector{<:AbstractVector}}
    ds::T # dimensionality of ContinuousTransition (dy, dx)
    f::F# transformation function
    es::V # unit vectors

    # NOTE: this meta is not user-friendly, I don't like a vector
    # perhaps making mutable struct with empty meta first will be better from user perspective
    # meta for transformation of a vector to a matrix
    function CTMeta(transformation::Function, â::Vector{<:Real})
        dy, dx = size(transformation(â))
        es = [StandardBasisVector(dy, i, one(eltype(first(â)))) for i in 1:dy]
        return new((dy, dx), transformation, es)
    end
end

getunits(meta::CTMeta)          = meta.es
getdimensionality(meta::CTMeta) = meta.ds
gettransformation(meta::CTMeta) = meta.f

getjacobians(ctmeta::CTMeta, a) = process_Fs(gettransformation(ctmeta), a)
process_Fs(f::Function, a) = [ForwardDiff.jacobian(a -> f(a)[i, :], a) for i in 1:size(f(a), 1)]

default_meta(::Type{CTMeta}) = error("ContinuousTransition node requires meta flag explicitly specified")

default_functional_dependencies_pipeline(::Type{<:ContinuousTransition}) = RequireMarginalFunctionalDependencies((3,), (nothing,))

"""
    `ctcompanion_matrix` casts a vector `a` into a matrix `A` by means of linearization of the transformation function `f` around the expansion point `a0`.
"""
function ctcompanion_matrix(a, epsilon, meta::CTMeta)
    a0 = a .+ epsilon # expansion point
    Js, es = getjacobians(meta, a0), getunits(meta)
    f = gettransformation(meta)
    dy, _ = getdimensionality(meta)
    # we approximate each row of A by a linear function and create a matrix A composed of the approximated rows
    A = sum(es[i] * (f(a0)[i, :] + Js[i] * (a - a0))' for i in 1:dy)
    return A
end

@average_energy ContinuousTransition (q_y_x::Any, q_a::Any, q_W::Any, meta::CTMeta) = begin
    ma, Va   = mean_cov(q_a)
    myx, Vyx = mean_cov(q_y_x)
    mW       = mean(q_W)

    dy, dx = getdimensionality(meta)
    Fs, es = getjacobians(meta, ma), getunits(meta)
    n      = div(ndims(q_y_x), 2)
    mA     = ctcompanion_matrix(ma, sqrt.(var(q_a)), meta)

    mx, Vx = @views myx[(dy + 1):end], Vyx[(dy + 1):end, (dy + 1):end]
    my, Vy = @views myx[1:dy], Vyx[1:dy, 1:dy]
    Vyx    = @view Vyx[1:dy, (dy + 1):end]
    g₁     = my' * mW * my + tr(Vy * mW)
    g₂     = mx' * mA' * mW * my + tr(Vyx * mA' * mW)
    g₃     = g₂
    G      = sum(sum(es[i]' * mW * es[j] * Fs[i] * (ma * ma' + Va) * Fs[j]' for i in 1:length(Fs)) for j in 1:length(Fs))
    g₄     = mx' * G * mx + tr(Vx * G)
    AE     = n / 2 * log2π - (mean(logdet, q_W) - (g₁ - g₂ - g₃ + g₄)) / 2

    return AE
end
