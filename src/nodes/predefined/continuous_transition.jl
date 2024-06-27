export CTransition, ContinuousTransition, CTMeta, ContinuousTransitionMeta

import LazyArrays
import StatsFuns: log2π

@doc raw"""
The ContinuousTransition node transforms an m-dimensional (dx) vector x into an n-dimensional (dy) vector y via a linear (or nonlinear) transformation with a `n×m`-dimensional matrix `A` that is constructed from a vector `a`.
ContinuousTransition node is primarily used in two regimes:

# When no structure on A is specified:
```julia
transformation = a -> reshape(a, 2, 2)
...
a ~ MvNormalMeanCovariance(zeros(2), Diagonal(ones(2)))
y ~ ContinuousTransition(x, a, W) where {meta = CTMeta(transformation)}
...
```
# When some structure if A is known:
```julia
transformation = a -> [cos(a[1]) -sin(a[1]); sin(a[1]) cos(a[1])]
...
a ~ MvNormalMeanCovariance(zeros(1), Diagonal(ones(1)))
y ~ ContinuousTransition(x, a, W) where {meta = CTMeta(transformation)}
...
```
To construct the matrix `A`, the elements of `a` are reshaped into `A` according to the transformation function provided in the meta. If you intend to use univariate Gaussian distributions, use it as a vector of length `1``, e.g. `a ~ MvNormalMeanCovariance([0.0], [1.;])`.

Check ContinuousTransitionMeta for more details on how to specify the transformation function that **must** return a matrix.

```julia
y ~ ContinuousTransition(x, a, W) where {meta = ContinuousTransitionMeta(transformation)}
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
`ContinuousTransitionMeta` is used as a metadata flag in `ContinuousTransition` to define the transformation function for constructing the matrix `A` from vector `a`.

`ContinuousTransitionMeta` requires a transformation function and the length of vector `a`, which acts as an expansion point for approximating the transformation linearly. If transformation appears to be linear, then no approximation is performed.

Constructors:
- `ContinuousTransitionMeta(transformation::Function, â::Vector{<:Real})`: Constructs a `ContinuousTransitionMeta` struct with the transformation function and allocated basis vectors.

Fields:
- `f`:  Represents the transformation function that transforms vector `a` into matrix `A`

The `ContinuousTransitionMeta` struct plays a pivotal role in defining how the vector `a` is transformed into the matrix `A`, thus influencing the behavior of the `ContinuousTransition` node.
"""
struct ContinuousTransitionMeta{F <: Function}
    f::F  # transformation function

    function ContinuousTransitionMeta(transformation::F) where {F}
        return new{F}(transformation)
    end
end

const CTMeta = ContinuousTransitionMeta

gettransformation(meta::CTMeta) = meta.f
# getctoutputdim(meta::CTMeta, J) = div(size(J, 2), size(J, 1)) # returns dy where J ∈ ℝ^{dx × dydx}

getjacobians(ctmeta::CTMeta, a) = process_Fs(gettransformation(ctmeta), a)
process_Fs(f::Function, a) = [ForwardDiff.jacobian(a -> f(a)[i, :], a) for i in 1:size(f(a), 1)]

default_meta(::Type{CTMeta}) = error("ContinuousTransition node requires meta flag explicitly specified")

default_functional_dependencies(::Type{<:ContinuousTransition}) = RequireMarginalFunctionalDependencies(a = nothing)

"""
    `ctcompanion_matrix` casts a vector `a` into a matrix `A` by means of linearization of the transformation function `f` around the expansion point `a0`.
"""
function ctcompanion_matrix(a, epsilon, meta::CTMeta)
    a0 = a + epsilon # expansion point
    Js = getjacobians(meta, a0)
    f  = gettransformation(meta)
    dy = length(Js)
    # we approximate each row of A by a linear function and create a matrix A composed of the approximated rows
    A = f(a0)
    for i in 1:dy
        A[i, :] .+= Js[i] * (a - a0)
    end
    return A
end

@average_energy ContinuousTransition (q_y_x::Any, q_a::Any, q_W::Any, meta::CTMeta) = begin
    ma, Va   = mean_cov(q_a)
    myx, Vyx = mean_cov(q_y_x)
    mW       = mean(q_W)

    Fs = getjacobians(meta, ma) # dx × dydx
    dy = length(Fs)

    n  = div(ndims(q_y_x), 2)
    mA = ctcompanion_matrix(ma, sqrt.(var(q_a)), meta)

    mx, Vx = @views myx[(dy + 1):end], Vyx[(dy + 1):end, (dy + 1):end]
    my, Vy = @views myx[1:dy], Vyx[1:dy, 1:dy]
    Vyx    = @view Vyx[1:dy, (dy + 1):end]

    g1 = -mA * Vyx'
    g2 = g1'
    trWSU, trkronxxWSU = zero(eltype(ma)), zero(eltype(ma))
    xxt = mx * mx'
    for (i, j) in Iterators.product(1:dy, 1:dy)
        FjVaFi = Fs[j] * Va * Fs[i]'
        trWSU += mW[j, i] * tr(FjVaFi)
        trkronxxWSU += mW[j, i] * tr(xxt * FjVaFi)
    end
    AE = n / 2 * log2π - mean(logdet, q_W) + (tr(mW * (mA * Vx * mA' + g1 + g2 + Vy + (mA * mx - my) * (mA * mx - my)')) + trWSU + trkronxxWSU) / 2

    return AE
end

@average_energy ContinuousTransition (q_y::Any, q_x::Any, q_W::Any, meta::CTMeta) = begin
    ma, Va = mean_cov(q_a)
    my, Vy = mean_cov(q_y)
    mx, Vx = mean_cov(q_x)
    mW = mean(q_W)

    Fs = getjacobians(meta, ma)
    dy = length(Fs)

    n = div(ndims(q_y), 2)
    mA = ctcompanion_matrix(ma, sqrt.(var(q_a)), meta)

    g1 = -mA
    g2 = g1'
    trWSU, trkronxxWSU = zero(eltype(ma)), zero(eltype(ma))
    xxt = mx * mx'
    for (i, j) in Iterators.product(1:dy, 1:dy)
        FjVaFi = Fs[j] * Va * Fs[i]'
        trWSU += mW[j, i] * tr(FjVaFi)
        trkronxxWSU += mW[j, i] * tr(xxt * FjVaFi)
    end
    AE = n / 2 * log2π - mean(logdet, q_W) + (tr(mW * (mA * Vx * mA' + g1 + g2 + Vy + (mA * mx - my) * (mA * mx - my)')) + trWSU + trkronxxWSU) / 2

    return AE
end
