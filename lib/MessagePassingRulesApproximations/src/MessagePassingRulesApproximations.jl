"""
    MessagePassingRulesApproximations

Numerics for propagating Gaussian moments through a function, and for expectations under a
normal. Everything here takes and returns means and covariances, plain numbers and arrays, never
distributions, and knows nothing of message passing: node packages, such as the Delta node's,
build their rules on it.

- [`Unscented`](@ref): the unscented transform, the mean and covariance of `f(x)` from `2d + 1`
  sigma points, through [`approximate`](@ref) and [`unscented_statistics`](@ref);
- [`Linearization`](@ref): the first-order expansion of `f` at the inputs' means, by automatic
  differentiation, through [`approximate`](@ref) and [`local_linearization`](@ref);
- [`GaussHermiteCubature`](@ref): expectations under a normal, and the moments of a normal
  reweighted by a function, through [`approximate_meancov`](@ref);
- [`smoothRTS`](@ref): the Rauch–Tung–Striebel correction of an input's marginal, from the
  forward statistics of either transform and a backward message.

# Examples

```jldoctest; setup = :(using MessagePassingRulesApproximations)
julia> m, V = approximate(Unscented(), x -> 2x + 1, (1.0,), (0.5,));

julia> m ≈ 3.0 && V ≈ 2.0
true

julia> approximate(Linearization(), x -> x^2, (3.0,))
(6.0, -9.0)
```
"""
module MessagePassingRulesApproximations

using LinearAlgebra
using FastCholesky: cholinv, cholsqrt
import ForwardDiff, FastGaussQuadrature

export AbstractApproximationMethod, approximation_name, approximation_short_name
export Unscented, UT, UnscentedTransform
export approximate, unscented_statistics, sigma_points_weights, smoothRTS
export Linearization, local_linearization
export GaussHermiteCubature, ghcubature, getweights, getpoints, approximate_meancov

include("approximations.jl")
include("shared.jl")
include("unscented.jl")
include("linearization.jl")
include("gausshermite.jl")
include("rts.jl")

end
